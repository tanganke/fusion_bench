"""
This module contains classes for representing tasks and examples as messages for chat-based interfaces.
"""

from abc import ABC, abstractmethod
from html import escape
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .arc import Example, Task
from .representers import (
    CompositeRepresenter,
    ConnectedComponentRepresenter,
    DelimitedGridRepresenter,
    DiffExampleRepresenter,
    GridRepresenter,
    ImageTaskRepresenter,
    PythonListGridRepresenter,
    TaskRepresenter,
    TextExampleRepresenter,
    TextTaskRepresenter,
    WordGridRepresenter,
)

MESSAGE = Dict[str, Union[str, Dict]]
MESSAGES = List[MESSAGE]


def display_messages(messages: MESSAGES):
    html_output = """<!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>Chat View</title>
    <style>
    /* CSS styling for chat interface */
    body {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    }
    .chat-container {
    width: 80%;
    max-width: 800px;
    margin: 0 auto;
    margin-top: 50px;
    }
    .message {
    display: block;
    clear: both;
    margin-bottom: 15px;
    }
    .message.user {
    text-align: right;
    }
    .message.assistant {
    text-align: left;
    }
    .message.system {
    text-align: left;
    }
    .message .bubble {
    display: inline-block;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 70%;
    position: relative;
    }
    .message.user .bubble {
    background-color: #0084ff;
    color: white;
    }
    .message.assistant .bubble {
    background-color: #e5e5ea;
    color: black;
    }
    .message.system .bubble {
    background-color: #e5e5ea;
    color: black;
    }
    .message .bubble img {
    max-width: 100%;
    border-radius: 10px;
    }
    .message .role {
    font-size: 0.8em;
    color: black;
    margin-bottom: 5px;
    }
    </style>
    </head>
    <body>
    <div class="chat-container">
    """

    # Loop through messages
    for message in messages:
        role = message.get("role", "user")
        content_list = message.get("content", [])
        if not content_list:
            continue  # Skip if no content
        if isinstance(content_list, str):
            content_list = [{"type": "text", "text": content_list}]

        # Start message div
        html_output += f'<div class="message {role}">\n'
        # Start bubble div
        html_output += '<div class="bubble">\n'
        # Add role label inside the bubble
        html_output += f'<div class="role">{role.capitalize()}</div>\n'

        # Process content items
        for content in content_list:
            content_type = content.get("type")
            if content_type == "text":
                text = content.get("text", "")
                # Escape HTML entities in text
                safe_text = escape(text)
                # Replace newlines with <br>
                safe_text = safe_text.replace("\n", "<br>")
                html_output += f"<p>{safe_text}</p>\n"
            elif content_type == "image_url":
                image_url = content["image_url"].get("url", {})
                if image_url:
                    html_output += f'<img src="{image_url}" alt="Image">\n'
            else:
                # Handle other content types if necessary
                pass

        # Close bubble and message divs
        html_output += "</div>\n</div>\n"

    # Close chat-container and body tags
    html_output += """
</div>
</body>
</html>"""

    return html_output


class MessageRepresenter(ABC):
    task_representer: TaskRepresenter

    @abstractmethod
    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        pass

    def display(self, messages: MESSAGES):
        return display_messages(messages)


# =============== MESSAGE REPRESENTATION ===============


class GPTTextMessagerepresenter(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[
            str
        ] = "Figure out the pattern in the following examples and apply it to the test case. {description}Your answer must follow the format of the examples. \n",
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        self.prompt = prompt
        self.task_representer = task_representer

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            desciption = "Here is a description of the task: \n\n{description}\n"
            description = desciption.format(description=task.description)
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        input_data.append({"role": "system", "content": prompt})

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(
                example, **kwargs
            )
            input_data.append({"role": "system", "content": query + output})

        query, output = self.task_representer.example_representer.encode(
            task.test_example, **kwargs
        )

        input_data.append({"role": "user", "content": query})

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError(
            "Decoding for GPTTextMessagerepresenter is not implemented."
        )


class GPTTextMessageRepresenterV2(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[
            str
        ] = "Figure out the underlying transformation in the following examples and apply it to the test case. {description}Here are some examples from this transformation, your answer must follow the format.\n",
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        self.prompt = prompt
        self.task_representer = task_representer
        # if example_representer is not None:
        #     self.task_representer.example_representer = example_representer(
        #                 io_sep=" -> ",
        #                 input_header="",
        #                 output_header="",
        #                 grid_representer=PythonListGridRepresenter
        #             )

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.task_representer.example_representer, DiffExampleRepresenter
        ):
            if self.task_representer.example_representer.use_output:
                prompt += "The input-diff-output grids are provided as python arrays where the diff is simply the output minus input:\n"
            else:
                prompt += "The input-diff grids are provided as python arrays:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.task_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals"
                if connected_component == 8
                else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided with indices of connected shapes ({connected_component}) of the same color:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer,
            PythonListGridRepresenter,
        ):
            prompt += "The input-output grids are provided as python arrays:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer,
            CompositeRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.task_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals"
                if connected_component == 8
                else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided as both python arrays and indices of connected shapes ({connected_component}) of the same color:\n"

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(
                example, **kwargs
            )
            if query is None or output is None:
                return None, None
            prompt += query + output + "\n"

        input_data.append({"role": "system", "content": prompt})

        query, output = self.task_representer.example_representer.encode(
            task.test_example, **kwargs
        )
        if query is None or output is None:
            return None, None

        input_data.append({"role": "user", "content": query})

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError(
            "Decoding for GPTTextMessageRepresenterV2 is not implemented."
        )

    def __repr__(self) -> str:
        return f"GPTTextMessageRepresenterV2(prompt={self.prompt!r}, task_representer={repr(self.task_representer)})"


class GPTTextMessageRepresenterV2CoT(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[str] = None,
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        if prompt:
            self.prompt = prompt
        else:
            self.prompt = "Figure out the underlying transformation in the following examples and apply it to the test case. {description}Here are some examples from this transformation, your answer must follow the format.\n"

        self.task_representer = task_representer

    def encode(self, task: Task) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.task_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            prompt += "The input-output grids are provided with indices of connected shapes of the same color:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer,
            PythonListGridRepresenter,
        ):
            prompt += "The input-output grids are provided as python arrays:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer,
            CompositeRepresenter,
        ):
            prompt += "The input-output grids are provided as both python arrays and indices of connected shapes of the same color:\n"

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(example)
            prompt += query + output + "\n"

        input_data.append({"role": "system", "content": prompt})

        query, output = self.task_representer.example_representer.encode(
            task.test_example
        )

        input_data.append(
            {"role": "user", "content": query + ". Let's think step by step:"}
        )

        cot_strs = ""
        for i, cot in enumerate(task.test_example.cot[:-1]):
            if -1 in cot:
                cot = np.where(cot == -1, 0, cot)
            cot_str = self.task_representer.example_representer.grid_representer.encode(
                cot
            )
            cot_str = "Step-" + str(i + 1) + ":\n" + cot_str
            cot_strs += cot_str + "\n"

        cot_strs += (
            "Final Step:\n"
            + self.task_representer.example_representer.grid_representer.encode(
                task.test_example.cot[-1]
            )
        )

        output_data = {"role": "assistant", "content": cot_strs}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError(
            "Decoding for GPTTextMessageRepresenterV2CoT is not implemented."
        )


class DataToCodeTextrepresenter(MessageRepresenter):
    def __init__(
        self,
        task_representer: TaskRepresenter = TextTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying code that produces the following input-output grids:\n",
    ):
        self.prompt = prompt
        self.task_representer = task_representer

    def encode(self, task: Task, code: str) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        prompt = self.prompt

        input_data.append({"role": "system", "content": prompt})

        data_points = ""

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(example)
            data_points += query + output + "\n"

        input_data.append({"role": "user", "content": data_points})

        output_data = {"role": "assistant", "content": code}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        # Decoding logic for DataToCodeTextrepresenter is complex and depends on the specific encoding format.
        # This is a placeholder for the actual implementation.
        raise NotImplementedError(
            "Decoding for DataToCodeTextrepresenter is not implemented."
        )


class GPTTextMessageRepresenterFewShot(MessageRepresenter):
    def __init__(
        self,
        task_representer: TaskRepresenter = TextTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
    ):
        self.prompt = prompt
        self.task_representer = task_representer

    def encode(
        self, task: Task, examples: List[Task], num_demonstrations: List[int]
    ) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        prompts = []
        for i, demo_task in enumerate(examples):
            k = num_demonstrations[i]
            if k >= 3:
                prompt = "== START OF TASK ==\n\n"
                demonstrations = demo_task.train_examples + [demo_task.test_example]
                for j in range(k):
                    example = demonstrations[j]
                    query, output = self.task_representer.example_representer.encode(
                        example
                    )
                    prompt += query + output + "\n\n"
                prompt += "== END OF TASK ==\n\n"
                prompts.append(prompt)

        prompts = "".join(prompts)

        input_data.append({"role": "system", "content": self.prompt + prompts})

        prompt = ""
        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(example)
            prompt += query + output + "\n\n"

        query, output = self.task_representer.example_representer.encode(
            task.test_example
        )

        input_data.append({"role": "user", "content": prompt + query})

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        # Decoding logic for GPTTextMessagerepresenterFewShot is complex and depends on the specific encoding format.
        # This is a placeholder for the actual implementation.
        raise NotImplementedError(
            "Decoding for GPTTextMessagerepresenterFewShot is not implemented."
        )


class GPTTextImageMessagerepresenter(MessageRepresenter):
    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformation in the following examples and apply it to the test case. {description}Here are some examples from this transformation, your answer must follow the format.\n",
    ):
        self.prompt = prompt
        self.text_representer = text_representer
        self.image_representer = image_representer

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.text_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals"
                if connected_component == 8
                else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        elif isinstance(
            self.text_representer.example_representer.grid_representer,
            PythonListGridRepresenter,
        ):
            prompt += "The input-output grids are provided both as image and as python arrays:\n"
        elif isinstance(
            self.text_representer.example_representer.grid_representer,
            CompositeRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals"
                if connected_component == 8
                else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."

        input_data.append({"role": "system", "content": prompt})

        for j, example in enumerate(task.train_examples + [task.test_example]):
            content = []
            query, output = self.text_representer.example_representer.encode(
                example, **kwargs
            )

            content.append(
                {
                    "type": "text",
                    "text": query.replace("\nOUTPUT:\n", ""),
                }
            )

            input_image = (
                self.image_representer.example_representer.grid_representer.encode(
                    example.input, **kwargs
                )
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                }
            )
            if j != len(task.train_examples):
                output_image = (
                    self.image_representer.example_representer.grid_representer.encode(
                        example.output, **kwargs
                    )
                )
                content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{output_image}"},
                    }
                )
            else:
                test_content = []
                output_image = (
                    self.image_representer.example_representer.grid_representer.encode(
                        example.output, **kwargs
                    )
                )
                test_content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

            input_data.append({"role": "user", "content": content})

        output_data = {
            "role": "assistant",
            "content": test_content,
        }

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError(
            "Decoding for GPTTextMessageRepresenterV2 is not implemented."
        )


class GPTTextImageMessageRepresenterFewShot(MessageRepresenter):
    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        diff_representer: Optional[GridRepresenter] = DelimitedGridRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
        disable_image: Optional[bool] = False,
        disable_text: Optional[bool] = False,
    ):
        self.prompt = prompt
        self.disable_image = disable_image
        self.disable_text = disable_text
        self.text_representer = text_representer
        self.image_representer = image_representer
        self.diff_representer = diff_representer

    def encode(
        self, task: Task, examples: List[Tuple[Task, str]]
    ) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        # if hasattr(task, "description"):
        #     description = task.description
        #     description = f"\n\n A possible description of the transformation: \n\n{description}\n"
        #     prompt = self.prompt.format(description=description)
        # else:
        #     prompt = self.prompt.format(description="")

        # if isinstance(self.text_representer.example_representer.grid_representer, ConnectedComponentRepresenter):
        #     connected_component = kwargs.get("connected_component", self.text_representer.example_representer.grid_representer.connected_component)
        #     connected_component = "including diagonals" if connected_component == 8 else "excluding diagonals"
        #     prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        # elif isinstance(self.text_representer.example_representer.grid_representer, PythonListGridRepresenter):
        #     prompt += "The input-output grids are provided both as image and as python arrays:\n"
        # elif isinstance(self.text_representer.example_representer.grid_representer, CompositeRepresenter):
        #     connected_component = kwargs.get("connected_component", self.text_representer.example_representer.grid_representer.connected_component)
        #     connected_component = "including diagonals" if connected_component == 8 else "excluding diagonals"
        #     prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."
        prompt = self.prompt
        input_data.append({"role": "system", "content": prompt})
        # Iterate over the examples provided for few-shot learning
        for example_task, example_output in examples:
            content = []
            for j, example in enumerate(
                example_task.train_examples + [example_task.test_example]
            ):
                query, output = self.text_representer.example_representer.encode(
                    example
                )
                if not self.disable_text:
                    content.append(
                        {
                            "type": "text",
                            "text": query.replace("\nOUTPUT:\n", ""),
                        }
                    )
                else:
                    content.append(
                        {
                            "type": "text",
                            "text": "\nINPUT:\n",
                        }
                    )

                if not self.disable_image:
                    input_image = self.image_representer.example_representer.grid_representer.encode(
                        example.input
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{input_image}"
                            },
                        }
                    )
                if j != len(example_task.train_examples):
                    if not self.disable_text:
                        content.append({"type": "text", "text": "\nOUTPUT:\n" + output})
                    else:
                        content.append({"type": "text", "text": "\nOUTPUT:\n"})
                    if not self.disable_image:
                        output_image = self.image_representer.example_representer.grid_representer.encode(
                            example.output
                        )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{output_image}"
                                },
                            }
                        )

                    if np.shape(example.input) == np.shape(example.output):
                        diff = example.output - example.input
                        diff = np.where(diff != 0, example.output, diff)
                        encoded_diff = self.diff_representer.encode(diff)
                        if not self.disable_text:
                            content.append(
                                {"type": "text", "text": "\nDIFF:\n" + encoded_diff}
                            )
                        else:
                            content.append({"type": "text", "text": "\nDIFF:\n"})

                        if not self.disable_image:
                            diff_image = self.image_representer.example_representer.grid_representer.encode(
                                diff
                            )
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{diff_image}"
                                    },
                                }
                            )

            input_data.append({"role": "user", "content": content})
            # reasoning
            input_data.append({"role": "assistant", "content": example_output})

        content = []
        for j, example in enumerate(task.train_examples + [task.test_example]):
            query, output = self.text_representer.example_representer.encode(example)
            if not self.disable_text:
                content.append(
                    {
                        "type": "text",
                        "text": query.replace("\nOUTPUT:\n", ""),
                    }
                )
            else:
                content.append(
                    {
                        "type": "text",
                        "text": "\nINPUT:\n",
                    }
                )
            if not self.disable_image:
                input_image = (
                    self.image_representer.example_representer.grid_representer.encode(
                        example.input
                    )
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                    }
                )

            if j != len(task.train_examples):
                if not self.disable_text:
                    content.append({"type": "text", "text": "\nOUTPUT:\n" + output})
                else:
                    content.append({"type": "text", "text": "\nOUTPUT:\n"})

                if not self.disable_image:
                    output_image = self.image_representer.example_representer.grid_representer.encode(
                        example.output
                    )

                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{output_image}"
                            },
                        }
                    )

                if np.shape(example.input) == np.shape(example.output):
                    diff = example.output - example.input
                    diff = np.where(diff != 0, example.output, diff)
                    encoded_diff = self.diff_representer.encode(diff)
                    if not self.disable_text:
                        content.append(
                            {"type": "text", "text": "\nDIFF:\n" + encoded_diff}
                        )
                    else:
                        content.append({"type": "text", "text": "\nDIFF:\n"})

                    if not self.disable_image:
                        diff_image = self.image_representer.example_representer.grid_representer.encode(
                            diff
                        )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{diff_image}"
                                },
                            }
                        )

        input_data.append({"role": "user", "content": content})

        output_data = [{}]

        return input_data, output_data


class TextMessageRepresenterFewShot(MessageRepresenter):
    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
    ):
        self.prompt = prompt
        self.text_representer = text_representer
        self.image_representer = image_representer

    def encode(
        self, task: Task, examples: List[Tuple[Task, str]], **kwargs
    ) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.text_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals"
                if connected_component == 8
                else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        elif isinstance(
            self.text_representer.example_representer.grid_representer,
            PythonListGridRepresenter,
        ):
            prompt += "The input-output grids are provided both as image and as python arrays:\n"
        elif isinstance(
            self.text_representer.example_representer.grid_representer,
            CompositeRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals"
                if connected_component == 8
                else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."

        input_data.append({"role": "system", "content": prompt})
        # Iterate over the examples provided for few-shot learning
        for example_task, example_output in examples:
            content = []
            for j, example in enumerate(
                example_task.train_examples + [example_task.test_example]
            ):
                query, output = self.text_representer.example_representer.encode(
                    example
                )

                content.append(
                    {
                        "type": "text",
                        "text": query.replace("\nOUTPUT:\n", ""),
                    }
                )

                if j != len(example_task.train_examples):
                    content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

            input_data.append({"role": "user", "content": content})
            # reasoning
            input_data.append({"role": "assistant", "content": example_output})

        content = []
        for j, example in enumerate(task.train_examples + [task.test_example]):
            query, output = self.text_representer.example_representer.encode(example)

            content.append(
                {
                    "type": "text",
                    "text": query.replace("\nOUTPUT:\n", ""),
                }
            )

            if j != len(task.train_examples):
                content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

        input_data.append({"role": "user", "content": content})

        output_data = [{}]

        return input_data, output_data


class GPTImageMessageRepresenterFewShot(MessageRepresenter):

    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
    ):
        self.prompt = prompt
        self.text_representer = text_representer
        self.image_representer = image_representer

    def encode(
        self, task: Task, examples: List[Tuple[Task, str]], **kwargs
    ) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.text_representer.example_representer.grid_representer,
            ConnectedComponentRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals"
                if connected_component == 8
                else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        elif isinstance(
            self.text_representer.example_representer.grid_representer,
            PythonListGridRepresenter,
        ):
            prompt += "The input-output grids are provided both as image and as python arrays:\n"
        elif isinstance(
            self.text_representer.example_representer.grid_representer,
            CompositeRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.text_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals"
                if connected_component == 8
                else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."

        input_data.append({"role": "system", "content": prompt})
        # Iterate over the examples provided for few-shot learning
        for example_task, example_output in examples:
            content = []
            for j, example in enumerate(
                example_task.train_examples + [example_task.test_example]
            ):

                query, output = self.text_representer.example_representer.encode(
                    example
                )

                input_image = (
                    self.image_representer.example_representer.grid_representer.encode(
                        example.input
                    )
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                    }
                )
                if j != len(example_task.train_examples):
                    output_image = self.image_representer.example_representer.grid_representer.encode(
                        example.output
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{output_image}"
                            },
                        }
                    )

            input_data.append({"role": "user", "content": content})
            # reasoning
            input_data.append({"role": "assistant", "content": example_output})

        content = []
        for j, example in enumerate(task.train_examples + [task.test_example]):
            query, output = self.text_representer.example_representer.encode(example)

            input_image = (
                self.image_representer.example_representer.grid_representer.encode(
                    example.input
                )
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                }
            )

            if j != len(task.train_examples):
                output_image = (
                    self.image_representer.example_representer.grid_representer.encode(
                        example.output
                    )
                )

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{output_image}"},
                    }
                )

        input_data.append({"role": "user", "content": content})

        output_data = [{}]

        return input_data, output_data


class GPTTextImageCodeMessageRepresenterFewShot(MessageRepresenter):
    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        image_representer: ImageTaskRepresenter = ImageTaskRepresenter(),
        prompt: Optional[
            str
        ] = "Figure out the underlying transformations in each task and complete the examples. You must follow the format.\n\n",
        disable_image: Optional[bool] = False,
    ):
        self.prompt = prompt
        self.disable_image = disable_image
        print("disable_image", self.disable_image)
        self.text_representer = text_representer
        self.image_representer = image_representer

    def encode(
        self, task: Task, task_reasoning: str, examples: List[Tuple[Task, str, str]]
    ) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        # if hasattr(task, "description"):
        #     description = task.description
        #     description = f"\n\n A possible description of the transformation: \n\n{description}\n"
        #     prompt = self.prompt.format(description=description)
        # else:
        #     prompt = self.prompt.format(description="")

        # if isinstance(self.text_representer.example_representer.grid_representer, ConnectedComponentRepresenter):
        #     connected_component = kwargs.get("connected_component", self.text_representer.example_representer.grid_representer.connected_component)
        #     connected_component = "including diagonals" if connected_component == 8 else "excluding diagonals"
        #     prompt += f"The input-output grids are provided with both as image and as indices of connected shapes ({connected_component}) of the same color."
        # elif isinstance(self.text_representer.example_representer.grid_representer, PythonListGridRepresenter):
        #     prompt += "The input-output grids are provided both as image and as python arrays:\n"
        # elif isinstance(self.text_representer.example_representer.grid_representer, CompositeRepresenter):
        #     connected_component = kwargs.get("connected_component", self.text_representer.example_representer.grid_representer.connected_component)
        #     connected_component = "including diagonals" if connected_component == 8 else "excluding diagonals"
        #     prompt += f"The input-output grids are provided as both python arrays and as indices of connected shapes ({connected_component}) of the same color."

        prompt = self.prompt

        input_data.append({"role": "system", "content": prompt})
        # Iterate over the examples provided for few-shot learning
        for example_task, reasoning, example_output in examples:
            content = []
            for j, example in enumerate(
                example_task.train_examples + [example_task.test_example]
            ):
                query, output = self.text_representer.example_representer.encode(
                    example
                )

                content.append(
                    {
                        "type": "text",
                        "text": query.replace("\nOUTPUT:\n", ""),
                    }
                )
                if not self.disable_image:
                    input_image = self.image_representer.example_representer.grid_representer.encode(
                        example.input
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{input_image}"
                            },
                        }
                    )
                if j != len(example_task.train_examples):

                    content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

                    if not self.disable_image:
                        output_image = self.image_representer.example_representer.grid_representer.encode(
                            example.output
                        )

                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{output_image}"
                                },
                            }
                        )

            input_data.append(
                {
                    "role": "user",
                    "content": content
                    + [
                        {
                            "type": "text",
                            "text": "\n\n====REASONING FOR CODE=====\n\n" + reasoning,
                        }
                    ],
                }
            )
            # reasoning
            input_data.append({"role": "assistant", "content": example_output})

        content = []
        for j, example in enumerate(task.train_examples + [task.test_example]):
            query, output = self.text_representer.example_representer.encode(example)

            content.append(
                {
                    "type": "text",
                    "text": query.replace("\nOUTPUT:\n", ""),
                }
            )
            if not self.disable_image:
                input_image = (
                    self.image_representer.example_representer.grid_representer.encode(
                        example.input
                    )
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{input_image}"},
                    }
                )

            if j != len(task.train_examples):
                content.append({"type": "text", "text": "\nOUTPUT:\n" + output})

                if not self.disable_image:
                    output_image = self.image_representer.example_representer.grid_representer.encode(
                        example.output
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{output_image}"
                            },
                        }
                    )

        input_data.append(
            {
                "role": "user",
                "content": content
                + [
                    {
                        "type": "text",
                        "text": "\n\n====REASONING FOR CODE=====\n\n" + task_reasoning,
                    }
                ],
            }
        )

        output_data = [{}]

        return input_data, output_data


class GPTCodeDebuggerMessager(MessageRepresenter):

    def __init__(
        self,
        text_representer: TextTaskRepresenter = TextTaskRepresenter(),
        prompt: str = "You are a debugging assistant. Please debug the code provided below.",
    ):
        self.prompt = prompt
        self.text_representer = text_representer

    def encode(self, task: Task, reasoning: str, code: str, error_message: str):
        # Prepare the input message for the model
        # system
        input_messages = [{"role": "system", "content": self.prompt}]

        demonstrations, query, output = self.text_representer.encode(task)
        query = demonstrations + "\n" + query.replace("\nOUTPUT:\n", "")

        content = []
        content.append(
            {
                "type": "text",
                "text": query
                + "\n\n====REASONING FOR CODE=====\n\n"
                + reasoning
                + "\n\n"
                + "```python\n"
                + code
                + "\n```",
            }
        )

        content.append(
            {
                "type": "text",
                "text": "\n\n Here is the error message:\n\n"
                + error_message
                + "\n\n Can you now give the debugged version of the code? Remember, the implementation must contain the ExampleRepresenter() class as that is used for testing. Do not make up your own Class names for the representer.",
            }
        )
        input_messages.append({"role": "user", "content": content})

        return input_messages, None


class GPTTextMessageRepresenterForBarc(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[
            str
        ] = "You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot patterns, and provide direct solutions.",
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        self.prompt = prompt
        self.task_representer = task_representer
        # if example_representer is not None:
        #     self.task_representer.example_representer = example_representer(
        #                 io_sep=" -> ",
        #                 input_header="",
        #                 output_header="",
        #                 grid_representer=PythonListGridRepresenter
        #             )

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        input_data.append({"role": "system", "content": self.prompt})

        content = "Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines.\nHere are the input and output grids for the reference examples:\n"

        for i, example in enumerate(task.train_examples):
            content += f"Example {i + 1}:\n"
            query, output = self.task_representer.example_representer.encode(
                example, **kwargs
            )
            if query is None or output is None:
                return None, None
            content += query + output + "\n"

        content += "\n\nHere is the input grid for the test example:\n"

        query, output = self.task_representer.example_representer.encode(
            task.test_example, **kwargs
        )

        query = query.replace("Output:", "")

        content += (
            query
            + "Directly provide the output grids corresponding to the given test input grids, based on the patterns observed in the reference examples."
        )

        input_data.append({"role": "user", "content": content})

        output = f"The output grid for the test input grid is:\n\n```\n{output}\n```"

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError(
            "Decoding for GPTTextMessageRepresenterV2 is not implemented."
        )

    def __repr__(self) -> str:
        return f"GPTTextMessageRepresenterForBarc(prompt={self.prompt!r}, task_representer={repr(self.task_representer)})"

    def __str__(self) -> str:
        return repr(self)


if __name__ == "__main__":
    print("Running tests")
    grid = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    example = Example(input=grid, output=grid)
    task = Task(test_example=example, train_examples=[example])

    representer = GPTTextMessageRepresenterForBarc(
        task_representer=TextTaskRepresenter(
            example_representer=TextExampleRepresenter(
                grid_representer=WordGridRepresenter(),
                input_header="Input:\n",
                output_header="\nOutput:\n",
                io_sep="\n",
            )
        )
    )

    input, output = representer.encode(task)
    breakpoint()

    representer = GPTTextMessagerepresenter()
    representer = GPTTextMessageRepresenterV2()
    breakpoint()
    input, output = representer.encode(task)
    print(input)
    html_output = representer.display(input)
    # Write to an HTML file
    with open("chat_view.html", "w", encoding="utf-8") as file:
        file.write(html_output)

    # representer = GPTTextMessageRepresenterV2(task_representer=TextTaskRepresenter(example_representer=TextExampleRepresenter(grid_representer=ConnectedComponentRepresenter())))

    representer = GPTTextImageMessagerepresenter()
    input, output = representer.encode(task)
    html_output = representer.display(input + [output])
    # Write to an HTML file
    with open("chat_view_w_image.html", "w", encoding="utf-8") as file:
        file.write(html_output)
