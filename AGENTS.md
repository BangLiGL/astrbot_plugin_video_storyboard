# Project maintenance

## Releases

The user requires every completed project update to be published to GitHub Releases as part of the same update. Treat each completed batch of changes, including documentation or icon changes, as a version update.

- Keep the version in `metadata.yaml`, README, CHANGELOG and the GitHub release tag consistent.
- Publish the release after the source commit is available and relevant checks have passed.
- Release assets must consist only of GitHub's automatically generated **Source code (zip)** and **Source code (tar.gz)** archives. Do not upload custom ZIP packages, images, binaries or other attachments to Releases.
- Write detailed Chinese release notes describing the concrete changes, resulting behavior, configuration or compatibility implications, and validation performed. Do not replace the notes with a vague one-line summary.
- Keep private server configuration, credentials, knowledge databases and test media out of the repository and release archives.

## Product behavior

- Video storyboards are internal evidence; final conversation belongs to AstrBot's current model and persona.
- Generate long-video notices only after resolving and injecting the currently effective AstrBot persona, including its preset dialogue. Skip notices if the persona cannot be confirmed.
- Retain the explicit division of message ownership with `astrbot_plugin_parser`.
- On the user's server, use existing cached media for actual tests; do not download new test videos or send test messages to real groups.
