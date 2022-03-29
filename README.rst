
**********************************************
**GLASS**: Generator for Large Scale Structure
**********************************************

.. image:: https://badges.gitter.im/glass-dev/glass.svg
   :alt: Join the chat at https://gitter.im/glass-dev/glass
   :target: https://gitter.im/glass-dev/glass?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

This is an early access repository for GLASS.

The best way to install the current code is to clone the repository and install
in develop mode via pip::

    git clone https://github.com/glass-dev/glass.git
    # or clone via ssh: git clone git@github.com:glass-dev/glass.git
    cd glass
    pip install -e .

You might want to have a look at the `documentation`__, as far as it exists at
this stage.  The `examples`__ page has some code to get you started.

__ https://glass.readthedocs.io/
__ https://glass.readthedocs.io/en/latest/examples/

If you want to run the examples yourself, you currently also need the `CAMB
module for GLASS`__::

    pip install git+https://github.com/glass-dev/glass-camb.git#egg=glass-camb

__ https://github.com/glass-dev/glass-camb

But there is probably no better way to get started than to get in touch.  Please
`join our Slack`__ and we can take it from there.

__ https://join.slack.com/t/glass-developers/shared_invite/zt-14s4x9qxz-r58swqSwmppyeE1fda6Zbw

(Note that the code currently does not have a license for distribution.  This is
done on purpose so that we can keep some vague notion of control over the code
before we have our v1.0 release and paper.  The code will be licensed under the
MIT license.)
