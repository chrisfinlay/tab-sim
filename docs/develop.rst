Developer Guide
===============

Merge features into main
------------------------

The main branch is protected from direct pushes. Changes to main can only be made via a pull request. tab-sim follows the Gitflow branching model. Therefore, there is a develop branch where features are branched from and then merged back to. Once a set of features are merged, a pull request is made back to main.

Since the default for merging into main is a squash merge, develop must then be rebased to main to get it back up to date to avoid future duplicate commits. To do this  

.. code-block:: bash

   git checkout develop
   git fetch origin
   git rebase origin/main
   git push --force-with-lease origin develop

Push a release
--------------

As part of merging develop into main, if the you intend to perform a release, the version in pyproject.toml must be updated as well. Assuming the merge to main has completed successfully you should 

.. code-block:: bash

   git checkout main
   git pull origin main
   git tag vX.X.X
   git push origin vX.X.X

Once this push is complete, a GitHub workflow will run to publish the release to PyPI.
