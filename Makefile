# this target runs checks on all files
quality:
	ruff check ochat
	ruff format --check ochat

# this target runs checks on all files and potentially modifies some of them
style:
	ruff check ochat --fix
	ruff format ochat