rd /s /q "build"
rd /s /q "dist"
python setup.py sdist bdist_wheel
twine upload -u __token__ -p pypi-AgENdGVzdC5weXBpLm9yZwIkMjhjN2JlZmItNGU3YS00N2JmLTk4ZTktMGI0YjJhZTdlODJjAAIleyJwZXJtaXNzaW9ucyI6ICJ1c2VyIiwgInZlcnNpb24iOiAxfQAABiDClTKKnPYFDG63IUQjyR7lfyDYCK5m29TRziCrByVUsg --repository-url https://test.pypi.org/legacy/ dist/* --verbose
