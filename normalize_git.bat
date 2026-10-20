@echo off
rem Normalize line endings using git attributes

if not exist .git goto no_git

echo Renormalizing line endings to LF/CRLF as specified in .gitattributes...
git add --renormalize .
echo Done! Staged normalized files. You can now commit them with:
echo   git commit -m "chore: normalize line endings using .gitattributes"
goto :eof

:no_git
echo Error: Not a git repository (no .git folder found).
exit /b 1
