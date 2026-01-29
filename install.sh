#!/usr/bin/env bash
# install.sh
# Skrypt instalacyjny dla FakeDetector (Debian)
# Uruchom w katalogu zawierającym fakedetector.py i default.conf (opcjonalnie).
#
# Funkcje:
# - opcjonalna instalacja zależności systemowych (apt)
# - tworzy virtualenv i instaluje pakiety z requirements.txt
# - kopiuje skrypt do /usr/local/bin/fakedetector (jeśli użytkownik wyrazi zgodę)
# - kopiuje default.conf do /etc/fakedetector/default.conf (jeśli istnieje i użytkownik wyrazi zgodę)
#
# Użycie:
#   sudo ./install.sh        # instalacja globalna (kopiuje do /usr/local/bin i /etc)
#   ./install.sh             # instalacja tylko do katalogu użytkownika (venv w $HOME)
#   ./install.sh --help

set -euo pipefail

REQ_FILE="requirements.txt"
SCRIPT_NAME="fakedetector.py"
DEFAULT_CONF="default.conf"
INSTALL_BIN="/usr/local/bin/fakedetector"
ETC_DIR="/etc/fakedetector"
VERBOSE=1

print() { echo -e "$@"; }
err() { echo "ERROR: $@" >&2; }

usage() {
  cat <<EOF
Użycie: $0 [--no-apt] [--venv DIR] [--yes]
 Opcje:
   --no-apt       Nie instaluj zależności systemowych przez apt (domyślnie pytamy).
   --venv DIR     Wskaż katalog dla virtualenv (domyślnie: jeśli uruchomiony jako root -> /opt/fakedetector/venv, inaczej -> $HOME/fakedetector-venv)
   --yes          Automatycznie potwierdź wszystkie operacje (nie pyta).
   --help         Pokaż pomoc.
EOF
}

ASK_YES=1
DO_APT=1
VENVDIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-apt) DO_APT=0; shift;;
    --venv) VENVDIR="$2"; shift 2;;
    --yes) ASK_YES=0; shift;;
    --help) usage; exit 0;;
    *) err "Nieznana opcja: $1"; usage; exit 2;;
  esac
done

confirm() {
  if [[ $ASK_YES -eq 0 ]]; then
    return 0
  fi
  read -p "$1 [y/N]: " -r REPLY
  case "$REPLY" in
    [Yy]* ) return 0;;
    * ) return 1;;
  esac
}

# Check that requirements and script exist
if [[ ! -f "$REQ_FILE" ]]; then
  err "Brak pliku $REQ_FILE w katalogu. Upewnij się, że uruchamiasz skrypt we właściwym katalogu."
  exit 3
fi
if [[ ! -f "$SCRIPT_NAME" ]]; then
  err "Brak pliku $SCRIPT_NAME w katalogu. Upewnij się, że skrypt fakedetector.py jest obecny."
  exit 3
fi

# Decide venv dir
if [[ -z "$VENVDIR" ]]; then
  if [[ $EUID -eq 0 ]]; then
    VENVDIR="/opt/fakedetector/venv"
  else
    VENVDIR="$HOME/fakedetector-venv"
  fi
fi

print "Virtualenv: $VENVDIR"
print "Requirements: $REQ_FILE"
print "Skrypt: $SCRIPT_NAME"

# Optional system install via apt (Debian)
if [[ $DO_APT -eq 1 ]]; then
  if command -v apt >/dev/null 2>&1; then
    if confirm "Czy chcesz zainstalować zależności systemowe (python3-venv, python3-pip, build-essential) przez apt?"; then
      print "Instalacja systemowych pakietów (wymaga sudo)..."
      sudo apt update
      sudo apt install -y python3 python3-venv python3-pip build-essential
    else
      print "Pominięto instalację pakietów systemowych."
    fi
  else
    print "APT nie jest dostępny — pomijam instalację systemowych pakietów."
  fi
fi

# Create venv
print "Tworzę virtualenv w: $VENVDIR"
mkdir -p "$(dirname "$VENVDIR")"
python3 -m venv "$VENVDIR"
# Activate and upgrade pip
# shellcheck source=/dev/null
source "$VENVDIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Install Python packages
print "Instaluję pakiety z $REQ_FILE w virtualenv..."
pip install -r "$REQ_FILE"

deactivate || true

# Optionally install executable and config (requires sudo)
if confirm "Czy skopiować skrypt do $INSTALL_BIN (wymaga sudo i umożliwi uruchamianie jako 'fakedetector')?"; then
  sudo cp "$SCRIPT_NAME" "$INSTALL_BIN"
  sudo chmod +x "$INSTALL_BIN"
  # Ensure shebang points to /usr/bin/env python3 (should be present in fakedetector.py)
  print "Skrypt skopiowany do $INSTALL_BIN"
else
  print "Pominięto kopiowanie skryptu do $INSTALL_BIN."
fi

# Copy default configuration if present
if [[ -f "$DEFAULT_CONF" ]]; then
  if confirm "Czy skopiować $DEFAULT_CONF do $ETC_DIR/default.conf (wymaga sudo)?"; then
    sudo mkdir -p "$ETC_DIR"
    sudo cp "$DEFAULT_CONF" "$ETC_DIR/default.conf"
    sudo chmod 644 "$ETC_DIR/default.conf"
    print "Skopiowano konfig do $ETC_DIR/default.conf"
  else
    print "Pominięto kopiowanie pliku konfiguracyjnego."
  fi
else
  print "Brak $DEFAULT_CONF w katalogu — nie kopiowano pliku konfiguracyjnego."
fi

# Final instructions
cat <<EOF

Instalacja zakończona.

Aby uruchomić FakeDetector:
 - Jeśli skopiowano do /usr/local/bin:
     fakedetector -i wiadomosc.txt -o wynik.json
 - Jeśli nie, aktywuj virtualenv i uruchom lokalnie:
     source "$VENVDIR/bin/activate"
     python "$PWD/$SCRIPT_NAME" -i wiadomosc.txt -o wynik.json
     deactivate

Plik konfiguracyjny domyślny (jeśli skopiowany):
  $ETC_DIR/default.conf

Jeśli chcesz, mogę przygotować instalację systemową (pakowanie do deb, unit systemd) lub zainstalować venv w /opt/ zamiast $VENVDIR.

EOF

exit 0
