import logging

# Define numeric values
TRACE = 5
NOTICE = 25

# Register level names so they show up in output and can be used in setLevel("TRACE")
logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(NOTICE, "NOTICE")


# Add convenience methods to Logger
def trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)


def notice(self, msg, *args, **kwargs):
    if self.isEnabledFor(NOTICE):
        self._log(NOTICE, msg, args, **kwargs)


logging.Logger.trace = trace
logging.Logger.notice = notice
