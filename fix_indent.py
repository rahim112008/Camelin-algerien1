"""
Corrige les blocs 'with tabs[N]:' dont le corps a perdu son indentation.
À exécuter : python fix_indent.py
"""
import re, sys
from pathlib import Path

SRC = Path("app.py")
TXT = SRC.read_text(encoding="utf-8")

lines = TXT.split("\n")
out = []
i = 0
fixes = 0
while i < len(lines):
    line = lines[i]
    m = re.match(r"^(\s*)with tabs\[\d+\]:\s*$", line)
    if m:
        base_indent = len(m.group(1))
        out.append(line)
        i += 1
        # Cherche le corps du bloc : toutes les lignes jusqu'à un 'with tabs[' au même niveau ou moins
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                out.append(nxt); i += 1; continue
            cur_indent = len(nxt) - len(nxt.lstrip())
            # Fin du bloc : un nouveau 'with tabs[' ou 'if __name__' au même niveau
            if re.match(r"^\s*with tabs\[\d+\]:", nxt) and cur_indent <= base_indent:
                break
            if re.match(r"^\s*if __name__", nxt) and cur_indent <= base_indent:
                break
            # Si la ligne n'est pas assez indentée (≤ base_indent), on la corrige
            if cur_indent <= base_indent:
                out.append(" " * (base_indent + 4) + nxt.lstrip())
                fixes += 1
            else:
                out.append(nxt)
            i += 1
    else:
        out.append(line)
        i += 1

NEW = "\n".join(out)
if fixes > 0:
    SRC.write_text(NEW, encoding="utf-8")
    print(f"✅ {fixes} ligne(s) corrigée(s) dans {SRC}")
else:
    print("ℹ️  Aucune correction nécessaire.")
