module.exports = {
  rules: {
    "header-max-length": [2, "always", 65],
    "subject-case": [0, "always", "sentence-case"],
    "scope-enum": [2, "always", ["all", "core", "fields", "galaxies",
                                 "lensing", "math", "observations", "points",
                                 "shapes", "shells", "user"]],
    "scope-case": [0, "always", "lower-case"],
    "type-enum": [2, "always", ["API", "BUG", "DEP", "DEV", "DOC", "ENH",
                                "MNT", "REV", "STY", "TST", "TYP", "REL"]],
    "type-case": [0, "always", "upper-case"],
  }
}
