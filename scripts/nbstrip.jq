# Git clean filter for .ipynb files: drops outputs and run-specific metadata so
# executing a notebook doesn't show up as a change. Setup is in the README.
(.cells[] | select(.cell_type == "code")) |= (.outputs = [] | .execution_count = null)
| .cells[].metadata |= del(.execution, .ExecuteTime, .collapsed, .scrolled)
| if .metadata.kernelspec then .metadata.kernelspec.display_name = "Python 3" else . end
| del(.metadata.language_info.version, .metadata.widgets, .metadata.vscode)
