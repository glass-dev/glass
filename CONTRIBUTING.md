# Contributing to GLASS

If you are planning to develop _GLASS_, or want to use the latest commit of
_GLASS_ on your local machine, you might want to install it from the source.
This installation is not recommended for users who want to use the stable
version of _GLASS_. The page below describes how to build, test, and develop
_GLASS_.

## Installation

The developer installation of _GLASS_ comes with several optional dependencies -

- `test`: installs extra packages used in tests and the relevant testing
  framework/plugins
- `docs`: installs documentation related dependencies
- `examples`: installs libraries used in the examples and a few notebook related
  dependencies

These options can be used with `pip` with the editable (`-e`) mode of
installation in the following way -

```bash
pip install -e ".[docs,test]"
```

## Tooling

### Pre-commit

_GLASS_ uses a set of `pre-commit` hooks and the `pre-commit.ci` bot to format,
lint, and prettify the codebase. The hooks can be installed locally using -

```bash
pre-commit install
```

This would run the checks every time a commit is created locally. The checks
will only run on the files modified by that commit, but the checks can be
triggered for all the files using -

```bash
pre-commit run --all-files
```

If you would like to skip the failing checks and push the code for further
discussion, use the `--no-verify` option with `git commit`.

## Testing

_GLASS_ is tested using `pytest` and `pytest-doctestplus`. `pytest` is
responsible for testing the code, whose configuration is available in
[pyproject.toml](https://github.com/glass-dev/glass/blob/main/pyproject.toml).
`pytest-doctestplus` is responsible for testing the examples available in every
docstring, which prevents them from going stale. Additionally, _GLASS_ also uses
`pytest-cov` (and [Coveralls](https://coveralls.io)) to calculate/display the
coverage of these unit tests.

### Running tests locally

The tests can be executed using the `test` dependencies of _GLASS_ in the
following way -

```bash
python -m pytest --cov=glass --doctest-plus
```

## Documenting

_GLASS_'s documentation is mainly written in the form of
[docstrings](https://peps.python.org/pep-0257) and
[reStructurredText](https://docutils.sourceforge.io/docs/user/rst/quickref.html).
The docstrings include the description, arguments, examples, return values, and
attributes of a class or a function, and the `.rst` files enable us to render
this documentation on _GLASS_'s documentation website.

_GLASS_ primarily uses [Sphinx](https://www.sphinx-doc.org/en/master/) for
rendering documentation on its website. The configuration file (`conf.py`) for
`sphinx` can be found
[under the `docs` folder](https://github.com/glass-dev/glass/blob/main/docs/conf.py).
The documentation is deployed on <https://readthedocs.io>
[here](https://glass.readthedocs.io/en/latest/).

Ideally, with the addition of every new feature to _GLASS_, documentation should
be added using comments, docstrings, and `.rst` files.

### Building documentation locally

The documentation is located in the `docs` folder of the main repository. This
documentation can be generated using the `docs` dependencies of _GLASS_ in the
following way -

```bash
cd docs/
make clean
make html
```

The commands executed above will clean any existing documentation build and
create a new build under the `docs/_build` folder. You can view this build in
any browser by opening the `index.html` file.

## Releases

To release a new version of _GLASS_, there should be a commit titled
`REL: glass yyyy.mm` that includes the following changes:

- The changelog trailers since the last release are parsed into the
  [changelog](CHANGELOG.md) under a section titled `[yyyy.mm]  (DD Mon YYYY)`. A
  new link to the changeset is added at the bottom of the file.
- The [release notes](docs/manual/releases.rst) are updated with the new
  version. The release notes should translate the changelog entries into prose
  that can be understood by non-developer users of the code. If there are
  breaking changes, a release note should explain what the changes mean for
  existing code.

Once these changes are merged into the `main` branch, a new release with title
`glass yyyy.mm` should be created in the GitHub repository. The description of
the release should be a copy of its release note.

Creating the release will automatically start the build process that uploads
Python packages for the new version to PyPI.

If any _GLASS_ extension packages depend on the new release, new versions of
these packages should be produced as soon as the new release is published to
PyPI.

### Versioning

_GLASS_ follows [CalVer](https://calver.org). There is no difference between
releases that increment the year and releases that increment the month; in
particular, releases that increment the month may introduce breaking changes.
The version is generate dynamically through VCS using
[`hatch-vcs`](https://github.com/ofek/hatch-vcs).

The target is to have a new _GLASS_ release at the beginning of each month, as
long as there have been changes.

The current version number is automatically inferred from the last release (i.e.
git tag), subsequent unreleased commits, and local changes, if any.

## Nox

`GLASS` supports running various critical commands using
[nox](https://github.com/wntrblm/nox) to make them less intimidating for new
developers. All of these commands (or sessions in the language of `nox`) -
`lint`, `tests`, `coverage`, `doctests`, `docs`, and `build` - are defined in
[noxfile.py](https://github.com/glass-dev/glass/main/noxfile.py).

`nox` can be installed via `pip` using -

```bash
pip install nox
```

The default sessions (`lint` and `tests`) can be executed using -

```bash
nox
```

A particular session (for example `tests`) can be run with `nox` on all
supported Python versions using -

```bash
nox -s tests
```

Only `tests`, `coverage`, and the `doctests` session run on all supported Python
versions by default.

To specify a particular Python version (for example `3.11`), use the following
syntax -

```bash
nox -s tests-3.11
```

The following command can be used to deploy the docs on `localhost` -

```bash
nox -s docs -- serve
```

The `nox` environments created for each type of session on the first run is
saved under `.nox/` and reused by default.

## Contributing workflow

Every change to the repository should come out of an issue where the change is
discussed.

[Pull requests](#pull-requests) should always follow from the discussion in an
existing issue. The only exception are minor, obvious changes such as fixing
typos in the documentation.

The discussion in a pull request should only be about the low-level details of
its implementation. All high-level, conceptual discussion belongs to the issue
to which the pull request refers.

### Pull requests

Pull requests to the `main` branch should have titles of the following form:

    gh-<issue-number>: Subject line

The body of the pull request should contain a description of the changes, and
any relevant details or caveats of the implementation.

The pull request should not repeat or summarise the discussion of its associated
issue. Instead, it should link to the issue using git's so-called "trailers".
These are lines of the form `key: value` which are at the end of the pull
request description, separated from the message body by a blank line.

To generically refer to an issue without any further action, use `Refs` and one
or more GitHub issue numbers:

    Refs: #12
    Refs: #25, #65

To indicate that the pull request shall close an open issue, use `Closes` and a
single GitHub issue number:

    Closes: #17

Changelog entries are collected using the following trailers, and later parsed
into the [changelog](CHANGELOG.md) for the next release:

    Added: Some new feature
    Changed: Some change in existing functionality
    Deprecated: Some soon-to-be removed feature
    Removed: Some now removed feature
    Fixed: Some bug fix
    Security: Some vulnerability was fixed

You can use any of the other common git trailers. In particular, you can use
`Cc` to notify others of your pull request via their GitHub user names:

    Cc: @octocat
