"""
This module contains classes for representing ARC tasks, examples, and grids in different formats.
"""

import re
from abc import ABC, abstractmethod
from io import BytesIO
from types import WrapperDescriptorType
from typing import List, Optional, Text, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from scipy.ndimage import generate_binary_structure, label

from .arc import Example, Grid, Task
from .np_cache import np_lru_cache

# =============== CONSTANTS ===============
COLUMN_SEP = " "
ROW_SEP = "\n"
IO_SEP = "\n\n"
EXAMPLE_SEP = "\n\n"
TRAIN_HEADER = "==TRAIN==\n"
TRAIN_TEST_SEP = "\n\n"
TEST_HEADER = "==TEST==\n"

# =============== UTILS ===============


def parse_numpy_from_str(array_str: str) -> np.ndarray:
    """
    Parses a string representation of a 2D array into a NumPy ndarray.

    Parameters:
    - array_str (str): A string representation of a 2D array, where rows are separated by newlines.

    Returns:
    - np.ndarray: A NumPy array of type int8 representing the parsed 2D array.
    """
    try:
        # Remove the surrounding brackets from the string
        clean_str = array_str.replace("[", "").replace("]", "")

        # Split the cleaned string by whitespace to get individual elements and convert them to integers
        elements = list(map(int, clean_str.split()))

        # Determine the number of rows by counting the newline characters and adding one
        rows = array_str.count("\n") + 1

        # Calculate the number of columns by dividing the total number of elements by the number of rows
        cols = len(elements) // rows

        # Create the NumPy array with the determined shape and convert it to type int8
        array = np.array(elements).reshape((rows, cols)).astype(np.int8)

        return array
    except Exception as e:
        # Print the exception message and the original string for debugging purposes
        print(e)
        print(array_str)
        # Return a default 1x1 array with a zero element in case of an error
        # raise e
        return None


# =============== INTERFACE ===============


class GridRepresenter(ABC):
    @abstractmethod
    def encode(self, grid: Grid) -> str:
        pass

    @abstractmethod
    def decode(self, encoded_str: str, **kwargs) -> Grid:
        pass

    def display(self, encoded_str: str):
        print(self.decode(encoded_str))


class ExampleRepresenter(ABC):
    grid_representer: GridRepresenter

    @abstractmethod
    def encode(self, example: Example, **kwargs) -> Union[str, Tuple[str, str]]:
        pass

    @abstractmethod
    def decode(self, encoded: Tuple[str, str], **kwargs) -> Example:
        pass

    def display(self, encoded: Union[str, Tuple[str, str]]):
        if isinstance(encoded, str):
            print(encoded)
        else:
            print("\n".join(encoded))


class TaskRepresenter(ABC):
    example_representer: ExampleRepresenter

    @abstractmethod
    def encode(self, task: Task, **kwargs) -> Union[Tuple[str, str, str], str]:
        pass

    @abstractmethod
    def decode(self, encoded: Tuple[str, str], **kwargs) -> Task:
        pass

    def display(self, encoded: Union[str, Tuple[str, str]]):
        if isinstance(encoded, str):
            print(encoded)
        else:
            print("\n".join(encoded))


# =============== GRID REPRESENTATION ===============
class DelimitedGridRepresenter(GridRepresenter):
    def __init__(self, column_sep: str = " ", row_sep: str = "\n"):
        self.column_sep: str = column_sep
        self.row_sep: str = row_sep

    def encode(self, grid: Grid) -> str:
        output = ""
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                output += str(grid[i][j]) + self.column_sep
            output = output[:-1] + self.row_sep
        return output[: -len(self.row_sep)]

    def decode(self, encoded_str: str) -> Grid:
        rows = encoded_str.strip().split(self.row_sep)
        grid = [list(map(int, row.split(self.column_sep))) for row in rows]
        return np.array(grid)

    def __repr__(self) -> str:
        return f"DelimitedGridRepresenter(column_sep={self.column_sep!r}, row_sep={self.row_sep!r})"

    def __str__(self) -> str:
        return repr(self)


class PythonListGridRepresenter(GridRepresenter):
    def encode(self, grid: Grid) -> str:
        return str(grid)

    def decode(self, encoded_str: str) -> Grid:
        return parse_numpy_from_str(encoded_str)

    def __repr__(self) -> str:
        return "PythonListGridRepresenter()"

    def __str__(self) -> str:
        return repr(self)


# Used in BARC
class WordGridRepresenter(GridRepresenter):
    color_map = {
        0: "Black",
        1: "Blue",
        2: "Red",
        3: "Green",
        4: "Yellow",
        5: "Gray",
        6: "Pink",
        7: "Orange",
        8: "Purple",
        9: "Brown",
    }

    def __init__(self):
        self.inv_map = {v: k for k, v in self.color_map.items()}

    def encode(self, grid: Grid) -> str:
        output = ""
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                output += self.color_map[grid[i][j]] + " "
            output = output[:-1] + "\n"
        return output[:-1]

    def decode(self, encoded_str: str) -> Grid:
        rows = encoded_str.strip().split("\n")
        grid = [[self.inv_map[color] for color in row.split()] for row in rows]
        return np.array(grid)

    def __str__(self) -> str:
        return "WordGridRepresenter()"

    def __repr__(self) -> str:
        return "WordGridRepresenter()"


# This is adapted from Greenblat 2024
class ConnectedComponentRepresenter(GridRepresenter):
    normalized: bool = True
    max_token_per_color: Optional[int] = None
    disable_absolute: bool = False
    dotsafter: int = 4
    connected_component: int = 4
    sort_by_count: bool = False
    spreadsheet_col_labels: List[str] = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "AA",
        "AB",
        "AC",
        "AD",
        "AE",
        "AF",
    ]

    def __init__(
        self,
        normalized: bool = True,
        max_token_per_color: Optional[int] = None,
        disable_absolute: bool = False,
        sort_by_count: bool = False,
    ):
        self.normalized = normalized
        self.max_token_per_color = max_token_per_color
        self.disable_absolute = disable_absolute
        self.sort_by_count = sort_by_count

    def to_spreadsheet(self, i: int, j: int) -> str:
        try:
            out = f"{self.spreadsheet_col_labels[j]}{i+1}"
        except IndexError:
            print(i, j)
            raise
        return out

    def to_spreadsheet_with_dots(self, rows_cols: List[Tuple[int, int]]) -> str:
        row_cols_v = np.array(sorted(rows_cols, key=lambda x: (x[0], x[1])))
        running_str = ""
        idx = 0
        while idx < len(row_cols_v):
            r, c = row_cols_v[idx]
            count_in_a_row = 0
            for checking_idx, (n_r, n_c) in enumerate(row_cols_v[idx:]):
                if n_r == r and n_c == c + checking_idx:
                    count_in_a_row += 1
                else:
                    break
            if count_in_a_row > self.dotsafter:
                start = self.to_spreadsheet(r, c)
                c_end = c + count_in_a_row - 1
                assert np.array_equal(
                    row_cols_v[idx + count_in_a_row - 1], (r, c_end)
                ), (
                    row_cols_v[idx + count_in_a_row - 1],
                    (r, c_end),
                )
                end = self.to_spreadsheet(r, c_end)
                running_str += f" {start} ... {end}"
                idx += count_in_a_row
            else:
                running_str += " " + self.to_spreadsheet(r, c)
                idx += 1
        return running_str

    def find_contiguous_shapes(self, grid: Grid, color: int) -> List[np.ndarray]:
        labeled_array, num_features = label(grid == color)
        shapes = []
        for i in range(1, num_features + 1):
            shapes.append(np.argwhere(labeled_array == i))
        if self.sort_by_count:
            shapes = sorted(shapes, key=lambda x: len(x), reverse=True)
        return shapes

    def find_contiguous_shapes_moore(self, grid: Grid, color: int) -> List[np.ndarray]:
        s = generate_binary_structure(2, 2)
        labeled_array, num_features = label(grid == color, structure=s)
        shapes = []
        for i in range(1, num_features + 1):
            shapes.append(np.argwhere(labeled_array == i))
        if self.sort_by_count:
            shapes = sorted(shapes, key=lambda x: len(x), reverse=True)
        return shapes

    def encode(self, grid: Grid, connected_component: Optional[int] = None) -> str:
        out = "[["
        if connected_component is None:
            connected_component = self.connected_component
        if connected_component == 4:
            out += "(4CC)\n"
        elif connected_component == 8:
            out += "(8CC)\n"

        color_shapes = []

        for color in range(10):
            if connected_component == 4:
                contiguous_shapes = self.find_contiguous_shapes(grid, color)
            elif connected_component == 8:
                contiguous_shapes = self.find_contiguous_shapes_moore(grid, color)
            color_shapes.append(contiguous_shapes)

        if self.sort_by_count:
            sorted_index = np.argsort(
                [
                    sum([len(shape) for shape in contiguous_shapes])
                    for contiguous_shapes in color_shapes
                ]
            )
        else:
            sorted_index = np.arange(10)

        for color in sorted_index:
            contiguous_shapes = color_shapes[color]
            if len(contiguous_shapes) == 0:
                continue
            shape_strings = []
            for shape in contiguous_shapes:
                if self.normalized:
                    min_i = min(i for i, j in shape)
                    min_j = min(j for i, j in shape)
                    normalized = [
                        (i - min_i, j - min_j)
                        for i, j in sorted(shape, key=lambda x: (int(x[0]), int(x[1])))
                    ]
                    basic_shape_str = self.to_spreadsheet_with_dots(normalized)
                    if not self.disable_absolute:
                        shape_str = (
                            "[Abs. "
                            + self.to_spreadsheet(min_i, min_j)
                            + "]"
                            + basic_shape_str
                        )
                    else:
                        shape_str = basic_shape_str
                else:
                    shape = [
                        (i, j)
                        for i, j in sorted(shape, key=lambda x: (int(x[0]), int(x[1])))
                    ]
                    shape_str = self.to_spreadsheet_with_dots(shape)
                shape_strings.append(shape_str)

            full_str = " | ".join(shape_strings)
            if self.max_token_per_color and self.max_token_per_color < len(
                full_str.split(" ")
            ):
                color_str = " [OMITTED DUE TO EXCESSIVE LENGTH]"
            else:
                color_str = full_str

            out += f"{color}: {color_str}\n"

        return out + "]]"

    def parse_position(self, pos):
        # find the letter part
        letter = re.findall(r"[A-Z]+", pos)[0]
        # find the number part
        number = re.findall(r"[0-9]+", pos)[0]
        row = int(number) - 1
        column = self.spreadsheet_col_labels.index(letter)
        return row, column

    def decode(self, encoded_str: str) -> Grid:
        encoded_str = encoded_str.replace("[[", "").replace("]]", "")
        encoded_str = encoded_str.replace("(8CC)\n", "").replace("(4CC)\n", "")
        max_row, max_col = (
            30,
            30,
        )  # Adjusted for the given example, can be adjusted if needed
        grid = np.full(
            (max_row, max_col), -1
        )  # Initialize with -1 to indicate empty cells

        # Process each color and its components
        for line in encoded_str.strip().split("\n"):
            color, components = line.split(": ")
            color = int(color)
            components = components.split(" | ")
            for component in components:
                component = component.replace("[Abs. ", "").replace("]", "")
                component = component.replace(" ... ", "...")
                abs_pos, *rel_positions = component.split()
                abs_pos = self.parse_position(abs_pos)
                abs_row, abs_col = abs_pos
                for rel_position in rel_positions:
                    if "..." in rel_position:
                        start_pos, end_pos = rel_position.split("...")
                        start_row, start_col = self.parse_position(start_pos)
                        end_row, end_col = self.parse_position(end_pos)
                        for row in range(start_row, end_row + 1):
                            grid[abs_row + row][
                                abs_col + start_col : abs_col + end_col + 1
                            ] = color
                    else:
                        rel_row, rel_col = self.parse_position(rel_position)
                        grid[abs_row + rel_row][abs_col + rel_col] = color

        # crop the grid from -1s
        rows = np.all(grid == -1, axis=1)
        cols = np.all(grid == -1, axis=0)
        grid = grid[~rows]
        grid = grid[:, ~cols]
        # replace remaining -1s with 0
        grid = np.where(grid == -1, 0, grid)
        return grid


class CompositeRepresenter(GridRepresenter):
    connected_component: int = 4

    def __init__(self, representers: List[GridRepresenter]):
        self.representers = representers

    def encode(
        self,
        grid: Grid,
        actives: Optional[List[int]] = None,
        connected_component: Optional[int] = None,
    ) -> str:
        if actives is not None:
            representers = [self.representers[i] for i in actives]
        else:
            representers = self.representers

        out = ""
        for representer in representers:
            if isinstance(representer, ConnectedComponentRepresenter):
                kwargs = {"connected_component": connected_component}
            else:
                kwargs = {}
            out += representer.encode(grid, **kwargs)
            out += "\n"
        return out.strip()

    def decode(self, encoded_str: str, actives: Optional[List[int]] = None) -> Grid:
        # Decoding logic for CompositeRepresenter is complex and depends on the specific encoding format.
        # This is a placeholder for the actual implementation.
        raise NotImplementedError(
            "Decoding for CompositeRepresenter is not implemented."
        )


class ImageGridRepresenter(GridRepresenter):
    cmap: ListedColormap = ListedColormap(
        [
            "#000000",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F012BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",
        ]
    )

    cnames: List[str] = [
        "black",
        "blue",
        "red",
        "green",
        "yellow",
        "gray",
        "magenta",
        "orange",
        "lightblue",
        "brown",
    ]

    @np_lru_cache(maxsize=8096)
    def encode(self, grid: Grid) -> str:
        # make sure the actual pixels are based on the grid's size
        fig, ax = plt.subplots(figsize=(len(grid[0]) / 2, len(grid) / 2))
        norm = Normalize(vmin=0, vmax=9)
        ax.imshow(grid, cmap=self.cmap, norm=norm)
        ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
        ax.set_yticks([x - 0.5 for x in range(1 + len(grid))])
        ax.set_xticks([x - 0.5 for x in range(1 + len(grid[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.tight_layout()
        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        # Encode the image in base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64

    def display(self, encoded_str: str):
        imgdata = base64.b64decode(encoded_str)
        plt.imshow(plt.imread(BytesIO(imgdata)))
        plt.axis("off")
        plt.show()

    def decode(self, encoded_str: str, **kwargs) -> Grid:
        raise NotImplementedError(
            "Decoding for ImageGridRepresenter is not implemented."
        )


class ConnectedComponentRepresenterV2(GridRepresenter):
    def __init__(self, sort_by_count: bool = False, connected_component: int = 4):
        self.sort_by_count = sort_by_count
        self.connected_component = connected_component

    def find_contiguous_shapes(
        self, grid: Grid, color: int, include_diagonals=False
    ) -> List[np.ndarray]:
        if include_diagonals:
            mask = generate_binary_structure(2, 2)
        else:
            mask = None
        labeled_array, num_features = label(grid == color, structure=mask)
        shapes = []
        for i in range(1, num_features + 1):
            shapes.append(np.argwhere(labeled_array == i))
        if self.sort_by_count:
            shapes = sorted(shapes, key=lambda x: len(x), reverse=True)
        return shapes

    def encode(self, grid: Grid, connected_component: Optional[int] = None) -> str:
        if connected_component is None:
            connected_component = self.connected_component

        color_shapes = []

        for color in range(10):
            contiguous_shapes = self.find_contiguous_shapes(
                grid, color, include_diagonals=connected_component == 8
            )
            color_shapes.append(contiguous_shapes)

        if self.sort_by_count:
            sorted_index = np.argsort(
                [
                    sum([len(shape) for shape in contiguous_shapes])
                    for contiguous_shapes in color_shapes
                ]
            )
        else:
            sorted_index = np.arange(10)
        # specify the shape
        output = f"(height={grid.shape[0]}, width={grid.shape[1]})\n"
        for k, color in enumerate(sorted_index):
            # skip color 0
            if color == 0:
                continue
            contiguous_shapes = color_shapes[color]
            if len(contiguous_shapes) == 0:
                continue
            shape_strings = []
            for shape in contiguous_shapes:
                min_i, min_j = np.min(shape, axis=0)
                max_i, max_j = np.max(shape, axis=0)

                subshape = grid[min_i : max_i + 1, min_j : max_j + 1]
                subshape_str = PythonListGridRepresenter().encode(subshape)

                shape_str = (
                    f"Shape(color={color}, pos=({min_i},{min_j}), grid={subshape_str})"
                )
                shape_strings.append(shape_str)

            output += "- " + "\n- ".join(shape_strings)
            if k != len(sorted_index) - 1:
                output += "\n"
        return output

    def decode(self, encoded_str: str) -> Grid:
        return None

    def __repr__(self) -> str:
        return f"ConnectedComponentRepresenterV2(sort_by_count={self.sort_by_count}, connected_component={self.connected_component})"


# =============== Example REPRESENTATION ===============


class TextExampleRepresenter(ExampleRepresenter):
    def __init__(
        self,
        io_sep: str = " -> ",
        input_header: str = "",
        output_header: str = "",
        output_footer="",
        grid_representer: GridRepresenter = PythonListGridRepresenter(),
    ):
        self.io_sep = io_sep
        self.input_header = input_header
        self.output_header = output_header
        self.output_footer = output_footer
        self.grid_representer = grid_representer

    def encode(self, example: Example, **kwargs) -> Tuple[str, str]:
        input_str = self.grid_representer.encode(example.input, **kwargs)
        if self.input_header:
            input_header = self.input_header
        else:
            input_header = ""

        output_str = self.grid_representer.encode(example.output, **kwargs)
        if self.output_header:
            output_header = self.output_header
        else:
            output_header = ""

        return (
            f"{input_header}{input_str}{self.io_sep}{output_header}",
            f"{output_str}{self.output_footer}",
        )

    def decode(self, encoded: Tuple[str, str], **kwargs) -> Example:
        input_str, output_str = encoded
        input_str = input_str.replace(self.input_header, "").replace(
            self.output_header, ""
        )
        if self.io_sep != "\n":
            input_str = input_str.replace(self.io_sep, "")
            input_str.strip()

        output_str = (
            output_str.replace(self.input_header, "")
            .replace(self.output_header, "")
            .strip()
        )

        input_grid = self.grid_representer.decode(input_str, **kwargs)
        output_grid = self.grid_representer.decode(output_str, **kwargs)
        return Example(input=input_grid, output=output_grid)

    def __repr__(self) -> str:
        return f"TextExampleRepresenter(io_sep={self.io_sep!r}, input_header={self.input_header!r}, output_header={self.output_header!r}, output_footer={self.output_footer!r}, grid_representer={repr(self.grid_representer)})"

    def __str__(self) -> str:
        return repr(self)


class ImageExampleRepresenter(ExampleRepresenter):
    def __init__(self, grid_representer: ImageGridRepresenter = ImageGridRepresenter()):
        self.grid_representer = grid_representer

    def encode(self, example: Example, **kwargs) -> str:
        input_grid = example.input
        output_grid = example.output

        # Create a figure with two subplots side by side
        # max height
        height = max(len(input_grid), len(output_grid))
        # max width
        width = max(len(input_grid[0]), len(output_grid[0]))
        fig, axes = plt.subplots(1, 2, figsize=(height, width))

        # Plot input grid
        grid = input_grid
        norm = Normalize(vmin=0, vmax=9)
        ax = axes[0]
        ax.imshow(input_grid, cmap=self.grid_representer.cmap, norm=norm)
        ax.set_title("Input")
        ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
        ax.set_yticks([x - 0.5 for x in range(1 + len(grid))])
        ax.set_xticks([x - 0.5 for x in range(1 + len(grid[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Plot output grid
        grid = output_grid
        ax = axes[1]
        ax.imshow(output_grid, cmap=self.grid_representer.cmap, norm=norm)
        ax.set_title("Output")
        ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
        ax.set_yticks([x - 0.5 for x in range(1 + len(grid))])
        ax.set_xticks([x - 0.5 for x in range(1 + len(grid[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        # Encode the image in base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64

    def display(self, encoded_str: str):
        imgdata = base64.b64decode(encoded_str)
        plt.imshow(plt.imread(BytesIO(imgdata)))
        plt.axis("off")
        plt.show()

    def decode(self, encoded_str: str, **kwargs) -> Example:
        raise NotImplementedError(
            "Decoding for ImageExampleRepresenter is not implemented."
        )


class DiffExampleRepresenter(ExampleRepresenter):
    def __init__(
        self,
        grid_representer: GridRepresenter = PythonListGridRepresenter(),
        io_sep: str = "\n\n",
        input_header: str = "INPUT:\n",
        output_header: str = "OUTPUT:\n",
        output_footer: str = "",
        use_output: bool = True,
        diff_sep: str = " + ",
        diff_output_sep: str = ",",
    ):
        self.io_sep = io_sep
        self.input_header = input_header
        self.output_header = output_header
        self.diff_output_sep = diff_output_sep
        self.diff_sep = diff_sep
        self.use_output = use_output
        self.output_footer = output_footer
        self.grid_representer = grid_representer

    def encode(self, example: Example, **kwargs) -> Tuple[str, str]:

        if np.shape(example.input) != np.shape(example.output):
            return None, None
        diff_str = self.grid_representer.encode(example.output - example.input)
        diff_str = diff_str.replace("  ", " ").replace("[[ ", "[[").replace("[ ", "[")
        input_str = self.grid_representer.encode(example.input, **kwargs)

        if self.use_output:
            output_str = self.grid_representer.encode(example.output, **kwargs)
            return (
                f"{self.input_header}{input_str}{self.io_sep}{self.output_header}",
                f"{self.diff_sep}{diff_str}{self.diff_output_sep}{output_str}{self.output_footer}",
            )
        else:
            return (
                f"{self.input_header}{input_str}{self.io_sep}{self.output_header}",
                f"{self.diff_sep}{diff_str}{self.output_footer}",
            )

    def decode(self, encoded: Tuple[str, str], **kwargs) -> Example:
        input_str, output_str = encoded
        input_str = (
            input_str.replace(self.input_header, "")
            .replace(self.output_header, "")
            .replace(self.io_sep, "")
            .strip()
        )
        output_str = (
            output_str.replace(self.diff_sep.strip(), "")
            .replace(self.output_footer, "")
            .strip()
        )
        input_grid = self.grid_representer.decode(input_str, **kwargs)
        if self.use_output:
            diff_str, output_str = output_str.split(self.diff_output_sep)
            output_grid = self.grid_representer.decode(output_str, **kwargs)
        else:
            diff_grid = self.grid_representer.decode(output_str, **kwargs)
            output_grid = input_grid + diff_grid
        return Example(input=input_grid, output=output_grid)

    def __repr__(self) -> str:
        return f"DiffExampleRepresenter(io_sep={self.io_sep!r}, input_header={self.input_header!r}, output_header={self.output_header!r}, output_footer={self.output_footer!r}, use_output={self.use_output}, diff_sep={self.diff_sep!r}, diff_output_sep={self.diff_output_sep!r}, grid_representer={repr(self.grid_representer)})"

    def __str__(self) -> str:
        return repr(self)


# =============== TASK REPRESENTATION ===============


class TextTaskRepresenter(TaskRepresenter):
    def __init__(
        self,
        train_header: str = "==TRAIN==\n",
        train_test_sep: str = "\n\n",
        test_header: str = "==TEST==\n",
        example_sep: str = "\n\n",
        example_representer: ExampleRepresenter = TextExampleRepresenter(),
    ):
        self.train_header = train_header
        self.train_test_sep = train_test_sep
        self.test_header = test_header
        self.example_sep = example_sep
        self.example_representer = example_representer

    def encode(self, task: Task, **kwargs) -> Tuple[str, str, str]:
        trains = self.train_header
        for train_example in task.train_examples:
            query, output = self.example_representer.encode(train_example, **kwargs)
            trains += query + output
            trains += self.example_sep

        trains = trains[: -len(self.example_sep)]

        demonstrations = trains

        test = self.test_header
        query, output = self.example_representer.encode(task.test_example, **kwargs)
        test += query

        return demonstrations, test, output

    def decode(self, encoded: Tuple[str, str, str], **kwargs) -> Task:
        train_examples = []
        demonstrations, test, encoded_output = encoded

        train_str = demonstrations.replace(self.train_header, "").strip()
        for example_str in train_str.split(
            self.example_sep + self.example_representer.input_header
        ):
            input_str, output_str = example_str.split(
                self.example_representer.io_sep + self.example_representer.output_header
            )
            train_example = self.example_representer.decode(
                (input_str, output_str), **kwargs
            )
            train_examples.append(train_example)

        test_input_str = test.replace(self.test_header, "")
        test_example = self.example_representer.decode(
            (test_input_str, encoded_output), **kwargs
        )

        return Task(train_examples=train_examples, test_example=test_example)

    def __repr__(self) -> str:
        return f"TextTaskRepresenter(train_header={self.train_header!r}, train_test_sep={self.train_test_sep!r}, test_header={self.test_header!r}, example_sep={self.example_sep!r}, example_representer={repr(self.example_representer)})"

    def __str__(self) -> str:
        return repr(self)


class ImageTaskRepresenter(TaskRepresenter):
    example_representer: ImageExampleRepresenter = ImageExampleRepresenter()

    def __init__(
        self, example_representer: Optional[ImageExampleRepresenter] = None
    ) -> None:
        if example_representer is not None:
            self.example_representer = example_representer

    def encode(self, task: Task, show_test_output=False, **kwargs) -> str:
        height = task.max_height()
        width = task.max_width()
        examples = task.train_examples + [task.test_example]
        height = len(examples)
        width = 2
        fig, axes = plt.subplots(len(examples), 2, figsize=(3 * height, 3 * width))
        norm = Normalize(vmin=0, vmax=9)
        for k, example in enumerate(examples):
            input_grid = example.input
            output_grid = example.output
            # Plot input grid
            grid = input_grid
            ax = axes[k, 0]
            ax.imshow(
                grid, cmap=self.example_representer.grid_representer.cmap, norm=norm
            )
            ax.set_title("Input")
            ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
            ax.set_yticks([x - 0.5 for x in range(1 + len(grid))])
            ax.set_xticks([x - 0.5 for x in range(1 + len(grid[0]))])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Plot output grid
            grid = output_grid
            ax = axes[k, 1]
            if not show_test_output and k == len(examples) - 1:
                # display black image
                grid = np.zeros((1, 1))
                ax.imshow(
                    grid, cmap=self.example_representer.grid_representer.cmap, norm=norm
                )
                ax.set_title("(Hidden)")
                ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
                ax.set_yticks([x - 0.5 for x in range(1 + len(grid))])
                ax.set_xticks([x - 0.5 for x in range(1 + len(grid[0]))])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax = axes[k, 1]
                ax.imshow(
                    grid, cmap=self.example_representer.grid_representer.cmap, norm=norm
                )
                ax.set_title("Output")
                ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
                ax.set_yticks([x - 0.5 for x in range(1 + len(grid))])
                ax.set_xticks([x - 0.5 for x in range(1 + len(grid[0]))])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        plt.tight_layout()

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        # Encode the image in base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64

    def display(self, encoded_str: str):
        imgdata = base64.b64decode(encoded_str)
        plt.imshow(plt.imread(BytesIO(imgdata)))
        plt.axis("off")
        plt.show()

    def decode(self, encoded_str: str, **kwargs) -> Example:
        raise NotImplementedError(
            "Decoding for ImageExampleRepresenter is not implemented."
        )


if __name__ == "__main__":

    example_representer = TextExampleRepresenter(
        grid_representer=WordGridRepresenter(),
        input_header="Input:\n",
        output_header="\nOutput:\n",
        io_sep="\n",
    )

    grid = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])

    example = Example(input=grid, output=grid)
    print(example_representer.encode(example))
    assert example == example_representer.decode(example_representer.encode(example))

    grid = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    representer = ConnectedComponentRepresenter()
    print(representer.encode(grid))
    assert np.array_equal(grid, representer.decode(representer.encode(grid)))
    representer.connected_component = 8
    print(representer.encode(grid))
    assert np.array_equal(grid, representer.decode(representer.encode(grid)))
    representer = DelimitedGridRepresenter()
    print(representer.encode(grid))
    assert np.array_equal(grid, representer.decode(representer.encode(grid)))
    representer = PythonListGridRepresenter()
    print(representer.encode(grid))
    assert np.array_equal(grid, representer.decode(representer.encode(grid)))
    representer = CompositeRepresenter(
        [DelimitedGridRepresenter(), ConnectedComponentRepresenter()]
    )
    print(representer.encode(grid))
    example = Example(input=grid, output=grid)
    representer = TextExampleRepresenter()
    print(representer.encode(example))
    assert example == representer.decode(representer.encode(example))
    task = Task(train_examples=[example, example], test_example=example)
    representer = TextTaskRepresenter()
    assert task == representer.decode(representer.encode(task))
    print(representer.encode(task))
    print(representer.display(representer.encode(task)))

    # image
    representer = ImageGridRepresenter()
    print(representer.encode(grid))
    # save as png
    base64_img = representer.encode(grid)
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(base64_img))

    representer = ImageExampleRepresenter()
    print(representer.encode(example))
    base64_img = representer.encode(Example(input=grid, output=grid))
    with open("example.png", "wb") as f:
        f.write(base64.b64decode(base64_img))

    representer = ImageTaskRepresenter()
    print(representer.encode(task))
    base64_img = representer.encode(task)
    with open("task.png", "wb") as f:
        f.write(base64.b64decode(base64_img))

    grid = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    representer = ConnectedComponentRepresenterV2()
    print(representer.encode(grid))

    representer = DiffExampleRepresenter(use_output=False)
    example = Example(input=grid, output=2 * grid)
    assert example == representer.decode(representer.encode(example))
    representer_str = repr(representer)
    print(representer_str)
