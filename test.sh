#!/bin/bash

# Standardwert festlegen
DEFAULT_VALUE=10

# Argument prüfen und entweder das angegebene Argument oder den Standardwert verwenden
NUMBER=${1:-$DEFAULT_VALUE}

# Ausgabe
echo "Die Zahl ist: $NUMBER"
