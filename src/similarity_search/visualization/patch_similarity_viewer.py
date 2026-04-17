"""Interactive Matplotlib viewer for patch-level similarity exploration.

This module provides UI logic to render patch overlays and handle click/keyboard
interaction for one-image and two-image similarity inspection workflows.
"""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from similarity_search.patch.grid import clamp_idx, idx_to_rc, rc_to_idx
from similarity_search.patch.math import upsample_nearest


class PatchSimilarityViewer:
    """Interactive patch-similarity viewer for one or two images.

    The viewer renders patch grids and cosine-similarity overlays, and supports
    click and keyboard interactions to inspect nearest-matching regions.
    """

    def __init__(
        self,
        states: list[dict[str, Any]],
        labels: list[str],
        show_grid: bool = False,
        show_overlay: bool = True,
        overlay_alpha: float = 0.55,
        annotate_indices: bool = False,
    ) -> None:
        """Initialize viewer state and display options.

        Args:
            states: One or two image states containing patch embeddings and metadata.
            labels: Display labels corresponding to states.
            show_grid: Whether patch grid lines are visible on startup.
            show_overlay: Whether similarity overlays are visible on startup.
            overlay_alpha: Overlay opacity in the range [0, 1].
            annotate_indices: Whether to draw per-patch index numbers.
        """
        if len(states) not in (1, 2):
            raise ValueError("PatchSimilarityViewer supports one or two images")
        if len(labels) != len(states):
            raise ValueError("labels length must match states length")

        self.states = states
        self.labels = labels
        self.show_grid = show_grid
        self.show_overlay = show_overlay
        self.overlay_alpha = overlay_alpha
        self.annotate_indices = annotate_indices

        self.two_image_mode = len(states) == 2
        self.cmap = plt.get_cmap("magma")

        self.active_side = 0
        self.current_idx = [
            (st["rows"] // 2) * st["cols"] + st["cols"] // 2 for st in states
        ]

        self.fig = None
        self.axs = None

    def _init_grid(self, ax, rows: int, cols: int, ps: int) -> list[plt.Line2D]:
        """Create grid line artists for patch boundaries on one axis."""
        grid = []
        for row in range(1, rows):
            line = ax.axhline(row * ps - 0.5, lw=0.8, alpha=0.6, color="white", zorder=3)
            grid.append(line)
        for col in range(1, cols):
            line = ax.axvline(col * ps - 0.5, lw=0.8, alpha=0.6, color="white", zorder=3)
            grid.append(line)
        return grid

    def _grid_set_visible(self, grid: list[plt.Line2D], is_visible: bool) -> None:
        """Toggle visibility for a list of grid line artists."""
        for line in grid:
            line.set_visible(is_visible)

    def _overlay_set_visible(self, ax_img: matplotlib.image.AxesImage, is_visible: bool) -> None:
        """Toggle visibility for one overlay image artist."""
        ax_img.set_visible(is_visible)

    def _draw_indices(self, ax, rows: int, cols: int, ps: int) -> None:
        """Draw patch index labels centered in each patch cell."""
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                ax.text(
                    col * ps + ps / 2,
                    row * ps + ps / 2,
                    str(idx),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    alpha=0.95,
                    zorder=4,
                )

    def _init_plot(self) -> None:
        """Create figure, axes, and initial artists for all configured states."""
        if self.two_image_mode:
            self.fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 11))
            self.axs = [ax_left, ax_right]
        else:
            self.fig, ax_left = plt.subplots(1, 1, figsize=(20, 11))
            self.axs = [ax_left]

        for ax, st in zip(self.axs, self.states):
            st["ax"] = ax
            ax.imshow(st["img"], zorder=0)
            ax.set_axis_off()

            st["grid"] = self._init_grid(ax, st["rows"], st["cols"], st["ps"])
            self._grid_set_visible(st["grid"], self.show_grid)

            if self.annotate_indices:
                self._draw_indices(ax, st["rows"], st["cols"], st["ps"])

            init_scalar = 0.5 * np.ones((st["rows"], st["cols"]), dtype=np.float32)
            rgba = self.cmap(init_scalar)
            rgba_up = upsample_nearest(rgba, st["h"], st["w"], st["ps"])
            st["overlay_im"] = ax.imshow(rgba_up, alpha=0.0, zorder=1)

            st["sel_rect"] = Rectangle((0, 0), st["ps"], st["ps"], fill=False, lw=2.0, ec="red", zorder=5)
            st["best_rect"] = Rectangle((0, 0), st["ps"], st["ps"], fill=False, lw=2.0, ec="yellow", zorder=6)
            ax.add_patch(st["sel_rect"])
            ax.add_patch(st["best_rect"])
            st["best_rect"].set_visible(False)

    def _set_titles(self, src_i: int | None = None, self_stats=None, cross_stats=None) -> None:
        """Update subplot titles and main figure title with interaction context."""
        if self.two_image_mode:
            self.axs[0].set_title(
                f"LEFT  {self.labels[0]} • {self.states[0]['rows']}x{self.states[0]['cols']} patches • "
                f"{'ACTIVE' if self.active_side == 0 else ''}",
                fontsize=10,
            )
            self.axs[1].set_title(
                f"RIGHT {self.labels[1]} • {self.states[1]['rows']}x{self.states[1]['cols']} patches • "
                f"{'ACTIVE' if self.active_side == 1 else ''}",
                fontsize=10,
            )
            if src_i is not None and self_stats is not None and cross_stats is not None:
                src_name = "LEFT" if src_i == 0 else "RIGHT"
                tgt_name = "RIGHT" if src_i == 0 else "LEFT"
                self.fig.suptitle(
                    f"Source: {src_name}  |  Self cos in [{self_stats[0]:.3f},{self_stats[1]:.3f}]  •  "
                    f"{tgt_name} cos in [{cross_stats[0]:.3f},{cross_stats[1]:.3f}]  |  "
                    f"Controls: click=select • arrows=move • '1'/'2'/'t'=switch side",
                    fontsize=11,
                )
            else:
                self.fig.suptitle(
                    "Controls: click=select • arrows=move • '1'/'2'/'t'=switch side",
                    fontsize=11,
                )
        else:
            self.axs[0].set_title(
                f"{self.labels[0]} • {self.states[0]['rows']}x{self.states[0]['cols']} patches",
                fontsize=10,
            )
            self.fig.suptitle("Controls: click=select • arrows=move", fontsize=11)

    def _update_selection_rects(self) -> None:
        """Move selection rectangles to current indices and update their visibility."""
        for idx, st in enumerate(self.states):
            row, col = idx_to_rc(self.current_idx[idx], st["cols"])
            st["sel_rect"].set_xy((col * st["ps"], row * st["ps"]))
        for idx, st in enumerate(self.states):
            st["sel_rect"].set_visible((not self.two_image_mode) or idx == self.active_side)

    def _compute_and_show_both_from_src(self, src_i: int) -> None:
        """Recompute and render self/cross similarity overlays from a source image."""
        src = self.states[src_i]
        q_idx = clamp_idx(self.current_idx[src_i], src["rows"], src["cols"])
        q = src["X"][q_idx]
        qn = q / (np.linalg.norm(q) + 1e-8)

        cos_self = np.matmul(src["Xn"], qn)
        cos_map_self = cos_self.reshape(src["rows"], src["cols"])
        disp_self = (cos_map_self - cos_map_self.min()) / (np.ptp(cos_map_self) + 1e-8)
        rgba_self = self.cmap(disp_self)

        row0, col0 = idx_to_rc(q_idx, src["cols"])
        rgba_self[row0, col0, 0:3] = np.array([1.0, 0.0, 0.0])
        rgba_self[row0, col0, 3] = 1.0

        src["overlay_im"].set_data(upsample_nearest(rgba_self, src["h"], src["w"], src["ps"]))
        src["overlay_im"].set_alpha(self.overlay_alpha)
        src["best_rect"].set_visible(False)

        if self.two_image_mode:
            tgt_i = 1 - src_i
            tgt = self.states[tgt_i]
            cos_cross = np.matmul(tgt["Xn"], qn)
            cos_map_cross = cos_cross.reshape(tgt["rows"], tgt["cols"])
            disp_cross = (cos_map_cross - cos_map_cross.min()) / (np.ptp(cos_map_cross) + 1e-8)
            rgba_cross = self.cmap(disp_cross)

            tgt["overlay_im"].set_data(
                upsample_nearest(rgba_cross, tgt["h"], tgt["w"], tgt["ps"])
            )
            tgt["overlay_im"].set_alpha(self.overlay_alpha)

            best = int(np.argmax(cos_cross))
            best_row, best_col = idx_to_rc(best, tgt["cols"])
            tgt["best_rect"].set_xy((best_col * tgt["ps"], best_row * tgt["ps"]))
            tgt["best_rect"].set_visible(True)
            self.states[src_i]["best_rect"].set_visible(False)

            self._set_titles(
                src_i,
                (cos_map_self.min(), cos_map_self.max()),
                (cos_map_cross.min(), cos_map_cross.max()),
            )
        else:
            self._set_titles(src_i, (cos_map_self.min(), cos_map_self.max()), None)

        self.fig.canvas.draw_idle()

    def _on_click(self, event) -> None:
        """Handle mouse clicks by selecting a patch and refreshing overlays."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        side = 0
        if self.two_image_mode:
            side = 0 if event.inaxes is self.axs[0] else (1 if event.inaxes is self.axs[1] else None)
            if side is None:
                return

        st = self.states[side]
        row = int(np.clip(event.ydata // st["ps"], 0, st["rows"] - 1))
        col = int(np.clip(event.xdata // st["ps"], 0, st["cols"] - 1))

        self.current_idx[side] = rc_to_idx(row, col, st["cols"])
        self.active_side = side
        self._update_selection_rects()
        self._compute_and_show_both_from_src(self.active_side)

    def _on_key(self, event) -> None:
        """Handle keyboard controls for navigation and display toggles."""
        side = self.active_side

        if self.two_image_mode:
            if event.key in ("t", "T"):
                self.active_side = 1 - self.active_side
                self._update_selection_rects()
                self._compute_and_show_both_from_src(self.active_side)
                return
            if event.key == "1":
                self.active_side = 0
                self._update_selection_rects()
                self._compute_and_show_both_from_src(self.active_side)
                return
            if event.key == "2":
                self.active_side = 1
                self._update_selection_rects()
                self._compute_and_show_both_from_src(self.active_side)
                return

        st = self.states[side]
        row, col = idx_to_rc(self.current_idx[side], st["cols"])

        if event.key in ("g", "G"):
            self.show_grid = not self.show_grid
            for state in self.states:
                self._grid_set_visible(state["grid"], self.show_grid)
        if event.key in ("o", "O"):
            self.show_overlay = not self.show_overlay
            for state in self.states:
                self._overlay_set_visible(state["overlay_im"], self.show_overlay)
        elif event.key in ("-", "_"):
            self.overlay_alpha = max(self.overlay_alpha - 0.05, 0.0)
        elif event.key in ("+", "="):
            self.overlay_alpha = min(self.overlay_alpha + 0.05, 1.0)
        elif event.key == "left":
            col = max(0, col - 1)
        elif event.key == "right":
            col = min(st["cols"] - 1, col + 1)
        elif event.key == "up":
            row = max(0, row - 1)
        elif event.key == "down":
            row = min(st["rows"] - 1, row + 1)
        else:
            return

        self.current_idx[side] = rc_to_idx(row, col, st["cols"])
        self._update_selection_rects()
        self._compute_and_show_both_from_src(self.active_side)

    def show(self) -> None:
        """Render the interactive viewer and register event callbacks."""
        self._init_plot()
        self._update_selection_rects()
        self._set_titles()
        self._compute_and_show_both_from_src(self.active_side)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        if self.two_image_mode:
            print("[two-image] Click to select • arrows move on ACTIVE side • '1'/'2'/'t' switch side")
        else:
            print("[single-image] Click to select • arrows move selection")

        plt.tight_layout()
        plt.show()
