.PHONY: format format_check static test test_coverage

format:
	black --line-length 100 .
	isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 100 src
