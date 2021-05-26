## Getting started

For more about how to use Jekyll, check out [this tutorial](https://www.taniarascia.com/make-a-static-website-with-jekyll/).
Why Jekyll? Read [Andrej Karpathy's blog post](https://karpathy.github.io/2014/07/01/switching-to-jekyll/)!


### Installation

#### Local setup

Assuming you have [Ruby](https://www.ruby-lang.org/en/downloads/) and [Bundler](https://bundler.io/) installed on your system (*hint: for ease of managing ruby gems, consider using [rbenv](https://github.com/rbenv/rbenv)*), first [fork](https://guides.github.com/activities/forking/) the theme from `github.com:alshedivat/al-folio` to `github.com:<your-username>/<your-repo-name>` and do the following:

```bash
$ git clone git@github.com:<your-username>/<your-repo-name>.git
$ cd <your-repo-name>
$ bundle install
$ bundle exec jekyll serve
```

Now, feel free to customize the theme however you like (don't forget to change the name!).
After you are done, **commit** your final changes.

#### Deployment

Deploying your website to [GitHub Pages](https://pages.github.com/) is the most popular option.
Starting version [v0.3.5](https://github.com/alshedivat/al-folio/releases/tag/v0.3.5), **al-folio** will automatically re-deploy your webpage each time you push new changes to your repository! :sparkles:

**For project pages (default):**

- Make changes, commit, and push!
- After deployment, the webpage will become available at `<your-github-username>.github.io/<your-repository-name>/`.
- The `master` branch should be used for the source code of your webpage and `gh-pages` branch (will be created on the first deployment) will be used for deployment.

**For personal and organization webpages:**
- Rename your repository to `<your-github-username>.github.io` or `<your-github-orgname>.github.io`.
- Click on **Actions** tab and **Enable GitHub Actions**; you no need to worry about creating any workflows as everything has already been set for you.
- In the **Settings**, select **Branches** and [rename the branch](https://docs.github.com/en/github/administering-a-repository/renaming-a-branch) with the source code from `master` to `source`. From now on, this will be your default branch. Any changes you make should be committed and pushed to this branch.
- Make sure the `url` and `baseurl` fields in `_config.yml` are empty.
- Make any other changes to your webpage, commit, and push. This will automatically trigger the **Deploy** action.
- Wait for a few minutes and let the action complete. You can see the progress in the **Actions** tab. If completed successfully, in addition to the `source` branch, your repository should now have a newly built `master` branch.
- Finally, again in the **Settings**, in the Pages section, set the branch to `master` (**NOT** to `source`).

**NOTE**: you **must** do all your changes in the `source` branch (the one you used to push) **NOT the master** one; this last one is used for **deploying** by Github Pages and it is not suitable for pushing changes.

