# dbm25

## Data set up
Download 'Data1.zip' and 'Data2.zip' from the university's Moodle page and extract them into the `data` folder. The data is not included in the repository due to its size.

## Start the devcontainer

```bash
- VSCode -> Show Command Palette (Ctrl+Shift+P)
- Dev Containers: Reopen in Container
```

## Start the Jupyter Notebook

This will start a Jupyter Notebook server in the devcontainer. You can access it at `http://localhost:8889` in your web browser.
```bash
jupyter notebook --allow-root --no-browser --port 8889
```
CAUTION: The `--allow-root` flag is necessary because the devcontainer runs as root. This is not recommended for production use, but is acceptable for testing purposes.
` From great powers comes great responsibility.`
