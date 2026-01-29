#!/usr/bin/env python3


fakedetector.py

Skrypt linii poleceń do podstawowej detekcji dezinformacji w tekście (polski).
- Przyjmuje plik wejściowy z treścią (-i)
- Zapisuje wynik JSON (-o)
- Wczytuje konfigurację YAML (/etc/fakedetector/default.conf lub -c)
- Używa lokalnych modeli Hugging Face jeśli skonfigurowano, w przeciwnym wypadku może
  opcjonalnie użyć OpenAI (jeśli podano klucz i włączono).

Uwaga:
- Skrypt stara się uruchamiać modele lokalnie. To wymaga zainstalowanych bibliotek
  (transformers, torch, sentence-transformers) i pobrania modeli wskazanych w konfiguracji.
- Jeśli lokalne modele nie są dostępne, fakt-checking może korzystać z OpenAI (opcjonalnie).

import argparse
import sys
import os
import json
import yaml
import logging
import math
import re
from collections import Counter
from typing import List, Dict, Any

# Optional imports (import lazily to keep startup light)
try:
    import torch
except Exception:
    torch = None

# We'll import transformer utilities only if needed at runtime.
# For keyword extraction we use sklearn's TfidfVectorizer
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

# Optional OpenAI client (import only if configured)
# Do not fail import here; will import when used.

LOG = logging.getLogger("fakedetector")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(
        prog="fakedetector",
        description="Detekcja dezinformacji w tekstach (polski).",
        add_help=False,
    )
    parser.add_argument("-i", "--input", dest="input_file", help="Plik z treścią (tekst).")
    parser.add_argument("-o", "--output", dest="output_file", help="Plik wyjściowy JSON.")
    parser.add_argument("-c", "--config", dest="config_file", help="Plik konfiguracyjny YAML.")
    parser.add_argument("-h", "--help", action="help", help="Pokaż tę pomoc i zakończ.")
    args = parser.parse_args()

    if not args.input_file or not args.output_file:
        parser.error("Musisz podać -i (plik wejściowy) i -o (plik wyjściowy).")
    return args

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_config(default_path: str, override_path: str = None) -> Dict[str, Any]:
    cfg = {}
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        LOG.warning("Plik konfiguracyjny domyślny nie istnieje: %s", default_path)

    if override_path:
        if os.path.exists(override_path):
            with open(override_path, "r", encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}
            cfg.update(override)
        else:
            LOG.warning("Plik konfiguracyjny nadpisujący nie istnieje: %s", override_path)

    return cfg

def preprocess_text(text: str) -> str:
    t = text.replace("\r\n", "\n").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def extract_keywords_tfidf(text: str, stopwords: List[str], top_n: int = 10) -> List[str]:
    if TfidfVectorizer is None:
        LOG.warning("sklearn nie jest zainstalowane; zwracam najczęstsze tokeny jako słowa kluczowe.")
        tokens = re.findall(r"\w{3,}", text.lower())
        tokens = [t for t in tokens if t not in (stopwords or [])]
        most = [w for w, _ in Counter(tokens).most_common(top_n)]
        return most

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=stopwords,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        X = vectorizer.fit_transform([text])
    except ValueError:
        return []
    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()[0]
    top_indices = tfidf_scores.argsort()[::-1][:top_n]
    keywords = [feature_array[i] for i in top_indices if tfidf_scores[i] > 0]
    return keywords

def detect_emotions_with_model(text: str, model_name: str, device: int = -1) -> List[str]:
    try:
        from transformers import pipeline

        p = pipeline("text-classification", model=model_name, return_all_scores=True, device=device)
        out = p(text[:1000])
        scores = out[0]
        scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)
        top = [s["label"] for s in scores_sorted[:5] if s["score"] > 0.05]
        return top
    except Exception as e:
        LOG.warning("Nie udało się uruchomić modelu emocji (%s): %s", model_name, e)
        return []

def detect_emotions_lexicon(text: str, lexicon: Dict[str, List[str]]) -> List[str]:
    text_low = text.lower()
    found = set()
    for emo, words in lexicon.items():
        for w in words:
            if re.search(r"\b" + re.escape(w) + r"\b", text_low):
                found.add(emo)
                break
    return list(found)

def detect_topic_zero_shot(text: str, candidate_labels: List[str], model_name: str, device: int = -1) -> str:
    try:
        from transformers import pipeline

        p = pipeline("zero-shot-classification", model=model_name, device=device)
        out = p(text[:1000], candidate_labels, multi_label=False)
        return out.get("labels", [candidate_labels[0]])[0]
    except Exception as e:
        LOG.warning("Nie udało się uruchomić zero-shot modelu (%s): %s", model_name, e)
        text_low = text.lower()
        for label in candidate_labels:
            for token in re.findall(r"\w+", label.lower()):
                if token in text_low:
                    return label
        return candidate_labels[0] if candidate_labels else "nieznany"

def fact_check_with_openai(text: str, openai_api_key: str, model_name: str = "gpt-4") -> Dict[str, Any]:
    try:
        import openai
    except Exception as e:
        LOG.error("Brak pakietu `openai`. Zainstaluj openai, jeśli chcesz korzystać z API OpenAI.")
        return {"conclusion": "unknown", "confidence": 0.5, "report": "Brak klienta OpenAI."}

    openai.api_key = openai_api_key
    prompt = (
        "Jesteś ekspertem weryfikacji faktów (fact-checker). Otrzymasz tekst w języku polskim."
        " Oceń, czy zawarte w nim twierdzenia są prawdziwe. "
        "Zwróć JSON z polskimi etykietami w następującym formacie:\n"
        '{ "conclusion": "true" | "false" | "partial", "confidence": 0.0-1.0, "report": "krótki raport 3-5 zdań po polsku" }\n'
        "Tekst (do weryfikacji):\n\n" + text
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.0,
        )
        content = resp["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            js = m.group(0)
            try:
                parsed = json.loads(js)
                return {
                    "conclusion": parsed.get("conclusion", "unknown"),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "report": parsed.get("report", "").strip(),
                }
            except Exception:
                pass
        c_lower = content.lower()
        conclusion = "unknown"
        if "fałsz" in c_lower or "nieprawda" in c_lower or "false" in c_lower:
            conclusion = "false"
        elif "prawda" in c_lower or "true" in c_lower:
            conclusion = "true"
        elif "częściowo" in c_lower or "częściowa" in c_lower or "partial" in c_lower:
            conclusion = "partial"
        conf_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", content)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        confidence = max(0.0, min(1.0, confidence))
        return {"conclusion": conclusion, "confidence": confidence, "report": content}
    except Exception as e:
        LOG.error("Błąd podczas wywołania OpenAI: %s", e)
        return {"conclusion": "unknown", "confidence": 0.5, "report": "Błąd przy kontakcie z API OpenAI."}

def fact_check_with_local_model(text: str, model_name: str, device: int = -1) -> Dict[str, Any]:
    try:
        from transformers import pipeline
    except Exception as e:
        LOG.warning("transformers nie jest zainstalowane: %s", e)
        return {"conclusion": "unknown", "confidence": 0.5, "report": "Brak biblioteki transformers."}

    pipeline_task = "text2text-generation"
    try:
        p = pipeline(pipeline_task, model=model_name, device=device)
    except Exception:
        try:
            p = pipeline("text-generation", model=model_name, device=device)
        except Exception as e:
            LOG.warning("Nie można uruchomić modelu fakt-check (%s): %s", model_name, e)
            return {"conclusion": "unknown", "confidence": 0.5, "report": "Nie udało się uruchomić lokalnego modelu."}

    prompt = (
        "Jesteś ekspertem weryfikacji faktów. Otrzymasz tekst w języku polskim. "
        "Oceń prawdziwość zawartych twierdzeń. Odpowiedz w języku polskim krótkim raportem (3-5 zdań), "
        "a na końcu dodaj ocenę w postaci: CONCLUSION: <true|false|partial>; CONFIDENCE: <0..1>\n\n"
        "Tekst:\n" + text
    )

    try:
        out = p(prompt, max_length=512, do_sample=False)
        gen = out[0].get("generated_text") or out[0].get("text") or ""
        m_conf = re.search(r"CONFIDENCE\s*[:\-]\s*([0-9]*\.?[0-9]+)", gen, re.I)
        m_conc = re.search(r"CONCLUSION\s*[:\-]\s*(true|false|partial)", gen, re.I)
        confidence = float(m_conf.group(1)) if m_conf else 0.5
        conclusion = m_conc.group(1).lower() if m_conc else "unknown"
        report = gen
        markers = re.search(r"CONCLUSION\s*[:\-]", gen, re.I)
        if markers:
            report = gen[: markers.start()].strip()
        return {"conclusion": conclusion, "confidence": max(0.0, min(1.0, confidence)), "report": report.strip()}
    except Exception as e:
        LOG.warning("Błąd podczas generowania odpowiedzi przez lokalny model: %s", e)
        return {"conclusion": "unknown", "confidence": 0.5, "report": "Błąd lokalnego modelu."}

def compute_fake_score(
    text: str,
    emotions: List[str],
    keywords: List[str],
    sensational_words: List[str],
    fact_check_result: Dict[str, Any],
    config: Dict[str, Any],
) -> float:
    text_low = text.lower()
    tokens = re.findall(r"\w{3,}", text_low)
    token_count = max(1, len(tokens))

    sens_count = 0
    for sw in sensational_words or []:
        sens_count += len(re.findall(r"\b" + re.escape(sw.lower()) + r"\b", text_low))
    sensational_score = min(1.0, sens_count / (token_count * 0.05 + 1e-9))  # scaled

    strong_emotions = set([e.lower() for e in config.get("strong_emotions", ["złość", "strach", "niepokój", "zaskoczenie", "oburzenie"])])
    emotion_score = 0.0
    if emotions:
        matches = sum(1 for e in emotions if e.lower() in strong_emotions)
        emotion_score = matches / max(1.0, len(emotions))

    hot_kw = config.get("hot_keywords", [])
    hot_count = sum(1 for kw in hot_kw if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text_low))
    hot_score = min(1.0, hot_count / max(1.0, len(hot_kw))) if hot_kw else 0.0

    fc_conf = float(fact_check_result.get("confidence", 0.5))
    fc_conc = fact_check_result.get("conclusion", "unknown")
    if fc_conc == "false":
        fc_score = 1.0 * (1.0)
    elif fc_conc == "partial":
        fc_score = 0.6
    elif fc_conc == "true":
        fc_score = 0.0
    else:
        fc_score = 0.5 * (1.0 - fc_conf)

    weights = config.get("scores_weights", {"sensational": 0.25, "emotion": 0.2, "hot": 0.15, "fact": 0.4})
    score = (
        weights.get("sensational", 0.25) * sensational_score
        + weights.get("emotion", 0.2) * emotion_score
        + weights.get("hot", 0.15) * hot_score
        + weights.get("fact", 0.4) * fc_score
    )

    score = max(0.0, min(1.0, score))
    score = math.floor(score * 100 + 0.5) / 100.0
    return score

def main():
    args = parse_args()
    default_conf_path = "/etc/fakedetector/default.conf"
    cfg = read_config(default_conf_path, args.config_file)

    stopwords = cfg.get("stopwords", ["i", "w", "z", "na", "do", "dla", "że", "się", "to", "jest", "o", "po"])
    tcfg = {
        "emotion_model": cfg.get("models", {}).get("emotion"),
        "zero_shot_model": cfg.get("models", {}).get("zero_shot"),
        "fact_model": cfg.get("models", {}).get("fact_check"),
        "use_openai": cfg.get("use_openai", False),
        "openai_api_key": cfg.get("openai_api_key"),
        "openai_model": cfg.get("openai_model", "gpt-4"),
        "keywords_top_n": cfg.get("keywords_top_n", 10),
        "sensational_words": cfg.get("sensational_words", ["sensacja", "szok", "pilne", "alarm", "koniec świata", "kataklizm"]),
        "candidate_topics": cfg.get("candidate_topics", ["polityka", "gospodarka", "zdrowie", "sport", "technologia", "prawo"]),
        "threshold_fakenews": cfg.get("threshold_fakenews", 0.7),
    }

    text = read_text_file(args.input_file)
    if not text:
        LOG.error("Plik wejściowy jest pusty: %s", args.input_file)
        sys.exit(2)
    text = preprocess_text(text)

    device = -1
    if torch is not None and torch.cuda.is_available():
        device = 0

    emotions = []
    if tcfg["emotion_model"]:
        emotions = detect_emotions_with_model(text, tcfg["emotion_model"], device=device)
    else:
        lex = {
            "złość": ["złość", "wściek", "oburzenie", "gniew"],
            "strach": ["strach", "przeraż", "panik"],
            "smutek": ["smut", "żał", "przykrość"],
            "radość": ["radość", "ciesz", "uśmiech"],
            "zaskoczenie": ["szok", "zaskocz", "sensacja"],
        }
        emotions = detect_emotions_lexicon(text, lex)

    topic = "nieznany"
    candidates = tcfg["candidate_topics"]
    if tcfg["zero_shot_model"]:
        topic = detect_topic_zero_shot(text, candidates, tcfg["zero_shot_model"], device=device)
    else:
        ltext = text.lower()
        found = None
        for cand in candidates:
            for token in re.findall(r"\w+", cand.lower()):
                if token and token in ltext:
                    found = cand
                    break
            if found:
                break
        topic = found or candidates[0] if candidates else "nieznany"

    keywords = extract_keywords_tfidf(text, stopwords, top_n=tcfg["keywords_top_n"])

    fact_result = {"conclusion": "unknown", "confidence": 0.5, "report": "Brak weryfikacji."}
    if tcfg["fact_model"]:
        LOG.info("Uruchamiam lokalny model fakt-check: %s", tcfg["fact_model"])
        fact_result = fact_check_with_local_model(text, tcfg["fact_model"], device=device)
    elif tcfg["use_openai"] and tcfg["openai_api_key"]:
        LOG.info("Używam OpenAI do weryfikacji faktów.")
        fact_result = fact_check_with_openai(text, tcfg["openai_api_key"], model_name=tcfg["openai_model"])
    else:
        LOG.info("Brak modelu fakt-check; weryfikacja ograniczona do heurystyk.")

    prob = compute_fake_score(
        text,
        emotions,
        keywords,
        tcfg["sensational_words"],
        fact_result,
        cfg,
    )
    is_fakenews = prob >= tcfg["threshold_fakenews"]

    opis = fact_result.get("report", "").strip()
    if not opis or opis.lower().startswith("brak"):
        opis = (
            "Analiza heurystyczna sugeruje, że część treści ma cechy sensacyjne i emocjonalne. "
            "Pełna weryfikacja faktów nie była możliwa lokalnie; zalecana jest dalsza kontrola "
            "przez źródła oficjalne."
        )

    output = {
        "lista_emocji": emotions,
        "temat": topic,
        "słowa_kluczowe": keywords,
        "prawdopodobieństwo": {
            "wartość": float(f"{prob:.2f}"),
            "fakenews": bool(is_fakenews),
        },
        "opis": opis,
        "_meta": {
            "fact_check": fact_result,
            "used_models": {
                "emotion_model": tcfg["emotion_model"],
                "zero_shot_model": tcfg["zero_shot_model"],
                "fact_model": tcfg["fact_model"],
                "openai_used": tcfg["use_openai"] and bool(tcfg["openai_api_key"]),
            },
        },
    }

    out_path = args.output_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    LOG.info("Wynik zapisano do: %s", out_path)
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()