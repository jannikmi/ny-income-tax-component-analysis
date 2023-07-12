
sync:
	#jupytext --sync notebooks/experiment.ipynb
	jupytext --sync scripts/experiment.py

update:
	@echo "pinning the dependencies specified in 'pyproject.toml':"
	poetry update

install:
	# for development all dependencies should be installed
	# --sync: remove everything not specified in the .lock file
	# --no-root: do not install the package itself
	poetry install --no-root --sync

lock:
	poetry lock --no-update -vvv


# when poetry dependency resolving gets stuck:
force_update:
	@echo "force updating the requirements. removing lock file"
	 poetry cache clear --all .
	 rm poetry.lock
	@echo "pinning the dependencies specified in 'pyproject.toml':"
	poetry update -vvv


outdated:
	poetry show --outdated



hook:
	pre-commit install
	pre-commit run --all-files

hookup:
	pre-commit autoupdate
