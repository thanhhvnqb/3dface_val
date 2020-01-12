"""Visualization."""
import matplotlib.patches as patches


def put_2dface(ax, points, face=None, color='w'):
    """Draw 2d poses."""
    if face is not None:
        rect = patches.Rectangle(
            (face[0], face[1]),
            face[2] - face[0],
            face[3] - face[1],
            linewidth=1,
            edgecolor='g',
            facecolor='none')
        ax.add_patch(rect)

    def plot(p_1, p_2):
        """Wrap for plot."""
        ax.plot(p_1, p_2, marker='o', markersize=4, linestyle='-', color=color)

    plot(points[0:17, 0], points[0:17, 1])
    plot(points[17:22, 0], points[17:22, 1])
    plot(points[22:27, 0], points[22:27, 1])
    plot(points[27:31, 0], points[27:31, 1])
    plot(points[31:36, 0], points[31:36, 1])
    plot(points[36:42, 0], points[36:42, 1])
    plot(points[[41, 36], 0], points[[41, 36], 1])
    plot(points[42:48, 0], points[42:48, 1])
    plot(points[[47, 42], 0], points[[47, 42], 1])
    plot(points[48:60, 0], points[48:60, 1])
    plot(points[[59, 48], 0], points[[59, 48], 1])
    plot(points[60:68, 0], points[60:68, 1])
    plot(points[[67, 60], 0], points[[67, 60], 1])


# def put_2dface(ax, face, points):
#     """Draw 2d poses."""
#     rect = patches.Rectangle(
#         (face[0], face[1]),
#         face[2] - face[0],
#         face[3] - face[1],
#         linewidth=1,
#         edgecolor='r',
#         facecolor='none')
#     ax.add_patch(rect)
#     ax.plot(points[0:17, 0], points[0:17, 1], marker='o', markersize=4, linestyle='-', color='w')
#     ax.plot(points[17:22, 0], points[17:22, 1], marker='o', markersize=4, linestyle='-', color='w')
#     ax.plot(points[22:27, 0], points[22:27, 1], marker='o', markersize=4, linestyle='-', color='w')
#     ax.plot(points[27:31, 0], points[27:31, 1], marker='o', markersize=4, linestyle='-', color='w')
#     ax.plot(points[31:36, 0], points[31:36, 1], marker='o', markersize=4, linestyle='-', color='w')
#     ax.plot(points[36:42, 0], points[36:42, 1], marker='o', markersize=4, linestyle='-', color='w')
#     ax.plot(
#         points[[41, 36], 0],
#         points[[41, 36], 1],
#         marker='o',
#         markersize=4,
#         linestyle='-',
#         color='w')
#     ax.plot(points[42:48, 0], points[42:48, 1], marker='o', markersize=4, linestyle='-', color='w')
#     ax.plot(
#         points[[47, 42], 0],
#         points[[47, 42], 1],
#         marker='o',
#         markersize=4,
#         linestyle='-',
#         color='w')
#     ax.plot(points[48:60, 0], points[48:60, 1], marker='o', markersize=4, linestyle='-', color='w')
#     ax.plot(
#         points[[59, 48], 0],
#         points[[59, 48], 1],
#         marker='o',
#         markersize=4,
#         linestyle='-',
#         color='w')
#     ax.plot(points[60:68, 0], points[60:68, 1], marker='o', markersize=4, linestyle='-', color='w')
#     ax.plot(
#         points[[67, 60], 0],
#         points[[67, 60], 1],
#         marker='o',
#         markersize=4,
#         linestyle='-',
#         color='w')
