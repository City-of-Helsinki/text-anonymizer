from text_anonymizer.config_cache import ConfigCache


def get_grant_list():
    return ConfigCache.instance().get_default_grantlist()


def get_block_list():
    return ConfigCache.instance().get_default_blocklist()
