Contributing guidelines
=======================

Committing
----------

Commits to the `main` branch should have messages that start with one of the
following prefixes to indicate the nature of the commit:

    API: an (incompatible) API change
    BUG: bug fix
    DEP: deprecate something, or remove a deprecated object
    DOC: documentation
    ENH: enhancement
    INF: project infrastructure (dev tools, packaging, etc.)
    MNT: maintenance commit (refactoring, typos, etc.)
    REV: revert an earlier commit
    STY: style fix (whitespace, PEP8)
    TST: addition or modification of tests
    TYP: static typing
    REL: related to releasing GLASS

Note that these apply to a commit that makes it into the `main` branch; for a
pull request on GitHub, that is the eventually squashed and merged commit.
The individual commits in a pull request can have arbitrary commit messages.

Pull requests on GitHub should have a label that matches the above prefixes, in
addition to any other applicable label (e.g. affected modules).


Versioning
----------

The target is to have a new *GLASS* release at the beginning of each month, as
long as there have been changes.

As soon as a new version has been released, the package information on the
`main` branch should be updated to the in-development version number
`yyyy.mm.dev0`.

Each breaking change should increment the in-development version number, e.g.
from `.dev0` to `.dev1`.  This is so that extension packages can catch up to
the core library at their own pace, and depend on the correct in-development
version.


Releasing
---------

To release a new version of *GLASS*, there should be a commit titled
`REL: glass yyyy.mm` that includes the following changes:

* The version of the `glass` core library is changed from `yyyy.mm.devN` to
  `yyyy.mm`.
* The current `Unreleased` section in the [changelog](CHANGELOG.md) is renamed
  to `yyyy.mm (DD Mon YYYY)` and a new "Unreleased" section is started.  The
  links to changesets at the bottom of the file have to be updated accordingly.
* The [release notes](docs/manual/releases.rst) are updated with the new
  version.  The release notes should translate the changelog entries into
  prose that can be understood by non-developer users of the code.  If there
  are breaking changes, a release note should explain what the changes mean for
  existing code.

Once these changes are merged into the `main` branch, a new release with title
`glass yyyy.mm` should be created in the GitHub repository.  The description of
the release should be a copy of its release note.

Creating the release will automatically start the build process that uploads
Python packages for the new version to PyPI.

Immediately after the release has been created, a new commit should increase
the minor version number and start a new development version (see
[versioning](#versioning)).

If any *GLASS* extension packages depend on the new release, new versions of
these packages should be produced as soon as the new release is published to
PyPI.
