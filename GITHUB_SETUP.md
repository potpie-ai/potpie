# GitHub Setup for Potpie Development

This repository has local GitHub authentication configured for pushing and creating PRs.

## Configuration

### Environment Variables (`.env`)
- `GH_TOKEN`: GitHub Personal Access Token for authentication

### Git Credential Helper

The repository uses a custom credential helper script that reads `GH_TOKEN` from environment:
- `.git-credentials-helper.sh`: Git credential helper for GitHub authentication

### Setup Details

The git config for this repo includes:
```
credential.helper=!/root/dev/potpie/.git-credentials-helper.sh
```

This allows git to use the `GH_TOKEN` environment variable for authentication without storing credentials in plain text.

## Usage

To push changes:
```bash
cd ~/dev/potpie
source .env  # Load GH_TOKEN
git push
```

To create PRs:
```bash
source .env
gh pr create --title "Your PR title" --body "Description"
```

## Token Scopes Required

For `gh` CLI operations (PR creation), the token needs:
- `repo` (Full control of private repositories)
- `read:org` (Read org and team membership)
- `read:discussion` (Read discussions)
- `read:project` (Read projects)

## To Update Token

1. Generate new token at https://github.com/settings/tokens
2. Update `.env` file:
   ```
   GH_TOKEN=your_new_token_here
   ```

## Security Notes

- Token is stored in `.env` file (not tracked by git - see .gitignore)
- Git credential helper reads token from environment at runtime
- No credentials are stored in git's credential cache or store
