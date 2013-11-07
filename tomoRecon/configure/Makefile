# Makefile

TOP=..

include $(TOP)/configure/CONFIG

# Set the following to NO to disable consistency checking of
# the support applications defined in $(TOP)/configure/RELEASE
CHECK_RELEASE = YES

TARGETS = $(CONFIG_TARGETS)
CONFIGS += $(subst ../,,$(wildcard $(CONFIG_INSTALLS)))

include $(TOP)/configure/RULES

