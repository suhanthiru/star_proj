import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial import KDTree

def launch_telescope(smooth_nodes, snapped_nodes, word_edges, sky_nodes, sky_ids, df_raw, anchor_coords, compass_dir, az_degrees):
    print("\n[+] Launching telescope viewfinder...")
    fig = plt.figure(figsize=(10, 10), facecolor='#020205') 
    ax = plt.gca()
    ax.set_facecolor('#020205')
    plt.title(f"Target: Look {compass_dir} ({az_degrees:.0f}°)", color='white', pad=20, fontsize=14, fontname='monospace')

    x_min, x_max = np.min(sky_nodes[:, 0]), np.max(sky_nodes[:, 0])
    y_min, y_max = np.min(sky_nodes[:, 1]), np.max(sky_nodes[:, 1])
    mw_num_points = 15000
    mw_x = np.random.uniform(x_min * 1.2, x_max * 1.2, mw_num_points)
    mw_y = mw_x * 0.6 + np.random.normal(0, (y_max - y_min) * 0.15, mw_num_points)
    ax.scatter(mw_x, mw_y, s=np.random.randint(100, 800, mw_num_points), c='#100a25', alpha=0.03, zorder=0)

    plt.scatter(sky_nodes[:, 0], sky_nodes[:, 1], s=1, color='#aaaaaa', alpha=0.3, zorder=1)

    constellation_x = smooth_nodes[:, 0]
    constellation_y = smooth_nodes[:, 1]
    
    beams = [ax.plot([smooth_nodes[e[0],0], smooth_nodes[e[1],0]], 
                     [smooth_nodes[e[0],1], smooth_nodes[e[1],1]], 
                     color='#ffffff', alpha=0.35, linewidth=2, zorder=2)[0] for e in word_edges]

    outer = ax.scatter(constellation_x, constellation_y, s=60, color='none', edgecolors='#ffffff', alpha=0.4, zorder=3)
    inner = ax.scatter(constellation_x, constellation_y, s=15, color='#ffffff', zorder=4)

    full_tree = KDTree(sky_nodes[:, :2])
    _, closest_indices = full_tree.query(snapped_nodes[:, :2])
    
    node_names = []
    for sid in sky_ids[closest_indices]:
        row = df_raw[df_raw['id'] == sid].iloc[0]
        proper, bf = str(row['proper']), str(row['bf'])
        if proper != "nan" and proper.strip(): node_names.append(proper)
        elif bf != "nan" and bf.strip(): node_names.append(bf)
        else: node_names.append(f"Star #{sid}")

    raw_stars = ax.scatter(sky_nodes[closest_indices, 0], sky_nodes[closest_indices, 1], s=80, color='#ffea00', alpha=0.9, zorder=5, visible=False)

    ax_btn = plt.axes([0.8, 0.05, 0.15, 0.05])
    btn = Button(ax_btn, 'Inspect Stars', color='#111122', hovercolor='#333344')
    btn.label.set_color('white')

    def on_toggle(event):
        state = not raw_stars.get_visible()
        raw_stars.set_visible(state)
        for b in beams: b.set_visible(not state)
        outer.set_alpha(0.05 if state else 0.4)
        inner.set_alpha(0.05 if state else 1.0)
        btn.label.set_text('Show Lines' if state else 'Inspect Stars')
        fig.canvas.draw_idle()
    btn.on_clicked(on_toggle)

    annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.5", fc="#0a0a1a", ec="#ffffff", alpha=0.9),
                        color="white", fontsize=10, fontname="monospace", zorder=10)
    annot.set_visible(False)
    aura = ax.scatter([], [], s=400, color='#00ffff', alpha=0.4, zorder=2)
    aura.set_visible(False)

    pan_state = {'is_panning': False, 'x': None, 'y': None}

    def zoom(event):
        base_scale = 1.2
        if event.xdata is None or event.ydata is None: return
        cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
        scale = 1/base_scale if event.button == 'up' else base_scale
        ax.set_xlim([event.xdata - (event.xdata - cur_xlim[0]) * scale, event.xdata + (cur_xlim[1] - event.xdata) * scale])
        ax.set_ylim([event.ydata - (event.ydata - cur_ylim[0]) * scale, event.ydata + (cur_ylim[1] - event.ydata) * scale])
        fig.canvas.draw_idle()

    def on_press(event):
        if event.button == 1 and event.inaxes and event.inaxes != ax_btn:
            pan_state.update({'is_panning': True, 'x': event.xdata, 'y': event.ydata})
            annot.set_visible(False)
            aura.set_visible(False)

    def on_release(event):
        if event.button == 1: pan_state['is_panning'] = False

    def on_mouse_move(event):
        if not event.inaxes or event.inaxes == ax_btn: return
        if pan_state['is_panning'] and event.xdata:
            dx, dy = event.xdata - pan_state['x'], event.ydata - pan_state['y']
            ax.set_xlim(ax.get_xlim()[0]-dx, ax.get_xlim()[1]-dx)
            ax.set_ylim(ax.get_ylim()[0]-dy, ax.get_ylim()[1]-dy)
            fig.canvas.draw_idle()
            return

        dists = np.sqrt((constellation_x - event.xdata)**2 + (constellation_y - event.ydata)**2)
        min_idx = np.argmin(dists)
        threshold = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02 
        
        if dists[min_idx] < threshold:
            aura.set_offsets(np.c_[constellation_x[min_idx], constellation_y[min_idx]])
            aura.set_visible(True)
            annot.xy = (constellation_x[min_idx], constellation_y[min_idx])
            annot.set_text(node_names[min_idx])
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                aura.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', zoom)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    cx, cy = np.mean(constellation_x), np.mean(constellation_y)
    zoom_radius = max(np.ptp(constellation_x), np.ptp(constellation_y)) * 0.6 + 10
    
    ax.set_xlim(cx - zoom_radius, cx + zoom_radius)
    ax.set_ylim(cy - zoom_radius, cy + zoom_radius)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    
    plt.show()