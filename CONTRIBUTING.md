Contributing guidelines
=======================

Workflow
--------

Every change to the repository should come out of an issue where the change is
discussed.

[Pull requests](#pull-requests) should always follow from the discussion in an
existing issue.  The only exception are minor, obvious changes such as fixing
typos in the documentation.

The discussion in a pull request should only be about the low-level details of
its implementation.  All high-level, conceptual discussion belongs to the issue
to which the pull request refers.


Pull requests
-------------

Pull requests to the `main` branch should have titles of the following form:

    TYPE: Subject line

The title can optionally refer to the module which is being changed:

    TYPE(module): Subject line

The `TYPE` prefix should indicate the nature of the change and must be taken
from the following list:

    API -- an (incompatible) API change
    BUG -- bug fix
    DEP -- deprecate something, or remove a deprecated object
    DEV -- development infrastructure (tools, packaging, etc.)
    DOC -- documentation
    ENH -- enhancement
    MNT -- maintenance commit (refactoring, typos, etc.)
    REV -- revert an earlier commit
    STY -- style fix (whitespace, PEP8)
    TST -- addition or modification of tests
    TYP -- static typing
    REL -- related to releasing GLASS

The optional `module` tag should indicate which modules are affected by the
change, and refer to an existing module name.

The body of the pull request should contain a description of the changes, and
any relevant details or caveats of the implementation.

The pull request should not repeat or summarise the discussion of its
associated issue.  Instead, it should link to the issue using git's so-called
"trailers".  These are lines of the form `key: value` which are at the end of
the pull request description, separated from the message body by a blank line.

To generically refer to an issue without any further action, use `Refs` and
one or more GitHub issue numbers:

    Refs: #12
    Refs: #25, #65

To indicate that the pull request shall close an open issue, use `Closes` and
a single GitHub issue number:

    Closes: #17

Changelog entries are collected using the following trailers, and later parsed
into the [changelog](CHANGELOG.md) for the next release:

    Added: Some new feature
    Changed: Some change in existing functionality
    Deprecated: Some soon-to-be removed feature
    Removed: Some now removed feature
    Fixed: Some bug fix
    Security: Some vulnerability was fixed

You can use any of the other common git trailers.  In particular, you can use
`Cc` to notify others of your pull request via their GitHub user names:

    Cc: @octocat


Versioning
----------

The target is to have a new *GLASS* release at the beginning of each month, as
long as there have been changes.

The current version number is automatically inferred from the last release
(i.e. git tag), subsequent unreleased commits, and local changes, if any.


Releasing
---------

To release a new version of *GLASS*, there should be a commit titled
`REL: glass yyyy.mm` that includes the following changes:

* The changelog trailers since the last release are parsed into the
  [changelog](CHANGELOG.md) under a section titled `[yyyy.mm]  (DD Mon YYYY)`.
  A new link to the changeset is added at the bottom of the file.
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

If any *GLASS* extension packages depend on the new release, new versions of
these packages should be produced as soon as the new release is published to
PyPI.
