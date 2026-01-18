import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logging.info("TEST 1")
logging.info("TEST 2")
logging.info("TEST 3")
print("If you see this print but not the logs above, logging is broken")
