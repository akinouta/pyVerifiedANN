class VO:
    def __init__(self, data, is_hash, level, site):
        self.data = data
        self.is_hash = is_hash
        self.level = level
        self.site = site

    def __str__(self):
        return f"{self.data}, is_hash:{self.is_hash}, level:{self.level}, site:{self.site}"
