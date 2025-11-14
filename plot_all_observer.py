import matplotlib.pyplot as plt


panel = 'v(s)'
dict_panel = {}
step_list = {}
for s in sorted_steps:
    for key in merged_by_step[s]:
        if key.startswith(panel + '/'):
            if key not in dict_panel:
                dict_panel[key] = []
                step_list[key] = []
            step_list[key].append(s)
            dict_panel[key].append(merged_by_step[s][key])
sorted_list_of_keys = sorted(dict_panel.keys(), key=lambda a: float(a.split('/')[1].split('m')[0]))

# Plot each key in a 3x4 grid of subfigures
subfigs = [3,4]
fig, axes = plt.subplots(subfigs[0], subfigs[1], figsize=(16, 10))
axes = axes.flatten()

for key in sorted_list_of_keys:
    idx = sorted_list_of_keys.index(key)
    if idx >= len(axes):
        # Safety: skip any keys beyond the grid capacity
        continue

    ax = axes[idx]
    x = step_list.get(key, [])
    y = dict_panel.get(key, [])

    # Plot series
    ax.plot(x, y, marker='.', linewidth=1, markersize=3)
    # Clean title: show milestone (e.g. "0.05m") and metric name after the slash
    title_parts = key.split('/')
    metric = title_parts[-1] if len(title_parts) > 1 else key
    ax.set_title(metric, fontsize=9)
    ax.grid(True)

    # Only label x ticks on the bottom row, and y ticks on the left column for readability
    row = idx // subfigs[1]
    col = idx % subfigs[1]
    if row < subfigs[0] - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("step")
    if col != 0:
        ax.set_yticklabels([])

# Hide any unused subplots
for j in range(len(sorted_list_of_keys), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle(f"Panel: {panel}", fontsize=14)
plt.show()