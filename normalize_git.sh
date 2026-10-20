#!/bin/bash
# Normalize line endings using git attributes

if [ ! -d ".git" ]; then
    echo "Error: Not a git repository (no .git folder found)."
    exit 1
fi

echo "Renormalizing line endings to LF/CRLF as specified in .gitattributes..."
git add --renormalize .
echo "Done! Staged normalized files. You can now commit them with:"
echo "  git commit -m \"chore: normalize line endings using .gitattributes\""
