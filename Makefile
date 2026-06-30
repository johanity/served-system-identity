# Reproduce the paper's numbers from the released logs. No API keys; standard library only.
#   make            reproduce everything (same as ./reproduce.sh)
#   make headline   one number at a time
.PHONY: all headline reproduction route_pin framing confab resample clean
all:
	@./reproduce.sh
headline reproduction route_pin framing confab resample:
	@cd reproduce && python3 $@.py
clean:
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} + ; echo "cleaned pycache"
