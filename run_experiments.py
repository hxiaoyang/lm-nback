import accelerate
import argparse
import copy
import gc
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from openai import OpenAI
import os
import re
import scipy.stats as stats
import seaborn as sns
from together import Together
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union


class PromptComposer:
    def __init__(
        self,
        version: str,
    ) -> None:
        self.version = version
        if version == "v1":
            self.template = "{} and {} are {}."
        if version == "v2":
            self.template = "$; current letter {} and letter # back {} are {}."

    def __call__(
        self,
        role: str,
        n: int,
        mode: str = None,
        cache: str = None,
    ) -> str:
        """
        Composes system prompt or correct assistant response.

        Args:
            role: If `system`, compose task instruction given `version`, `n` and
                `mode`. `cache` is irrelevant. If `assistant`, compose correct
                response given `version`, `n` and `cache`. `mode` is irrelevant.
            n: n-back task.
            mode: Either `demo` or `test`.
            cache: List containing current letter and previous n letters.
        Returns:
            System prompt or correct assistant response.
        """
        if role == "system":
            s1 = "" if n == 1 else "s"
            s2 = "first letter" if n == 1 else f"first {n} letters"
            s3 = "previous letter" if n == 1 else f"previous {n} letters, from new to old"
            s4 = "previous one" if n == 1 else f"one {n} step{s1} before"

            if mode == "demo":
                ret = (
                    "You are a participant in a cognitive task. "
                    "You will be shown a sequence of letters, one at a time. "
                )

            if mode == "test":
                ret = (
                    "Once again, you are a participant in a cognitive task. "
                    "You will be shown a new sequence of letters, one at a time. "
                )

            if self.version == "v1":
                ret += (
                    f"For each letter, determine if it is the same as the letter {n} step{s1} before it. "
                    "Answer in the following format: "
                    f"\"[current letter] and [letter {n} back] are [identical/different].\" "
                    f"Note that, for the {s2}, there won't be any letter {n} step{s1} back; "
                    f"write \"none\" for \"[letter {n} back]\" in this case. "
                )

            if self.version == "v2":
                ret += (
                    f"In each response, recall the current and {s3}. "
                    f"Then, determine if the current letter is the same as the {s4}. "
                    "Answer in the following format: \"current: [letter], "
                )
                ret += ", ".join(f"{i+1} back: [letter]" for i in range(n))
                ret += f"; current letter [letter] and letter {n} back [letter] are [identical/different].\" "
                ret += (
                    f"Note that, for the {s2}, there won't be any letter {n} step{s1} back; "
                    f"write \"none\" for \"[letter]\" in this case. "
                )

            if mode == "test":
                ret += "Remember, this is a new sequence. "

            ret += "Let's think step by step."
            return ret

        if role == "assistant":
            comparison = "identical" if cache[0] == cache[-1] else "different"
            ret = self.template.format(cache[0], cache[-1], comparison)
            f = lambda x: f"current: {x[0]}, " + ", ".join(f"{i+1} back: {y}" for i,y in enumerate(x[1:]))
            ret = ret.replace("#", str(n)).replace("$", f(cache))
            return ret


class NBackTest:
    def __init__(
        self,
        directory: str,
        version: str,
        model_name: str,
        use_api: bool = False,
        api_key: bool = None,
        use_model: bool = False,
        use_tokenizer: bool = False,
        use_auth_token: str = None,
        interactive: bool = False,
        curriculum: bool = False,
    ) -> None:
        self.directory = directory
        self.dialog = []
        self.version = version
        self.prompter = PromptComposer(version)
        self.interactive = interactive
        self.curriculum = curriculum

        self.model_dir = model_name.split("/")[0] if "/" in model_name else ""
        self.model_name = model_name.split("/")[-1]
        model_name_hf = model_name[:-6] if "Llama" in model_name else model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_api:
            self.client = OpenAI(api_key=api_key) if "gpt" in model_name else Together(api_key=api_key)

        if use_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_hf,
                use_auth_token=use_auth_token,
            )

        if use_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_hf,
                torch_dtype=torch.float16,
                device_map="auto",
                use_auth_token=use_auth_token,
            )

    def _fetch(
        self,
        data_type: str,
        n: int,
        m: int = -1,
        demo: int = 1,
        i: int = 0,
        j: int = 0,
        response_types: list[str] = ["past","past"],
    ) -> Union[np.ndarray, dict]:
        """
        Fetches task data or model outputs.
        """
        d1 = f"{self.directory}/data"
        d2 = f"{self.directory}/outputs"

        if data_type == "hidden_states" or data_type == "attentions":
            path = "{}/{}/{}back_{}_{}_#{}.npy".format(
                d2, data_type, n, self.model_name, self.version, j,
            )
            return np.load(path)

        if data_type == "task":
            path = "{}/{}back.json".format(d1, n)

        if data_type == "interactive_demos":
            path = "{}/{}back_interact.json".format(d1, n)

        if data_type == "generations":
            suffix = "_interactive" if self.interactive else "_curriculum" if self.curriculum else ""
            path = "{}/{}/{}back_{}_{}{}.json".format(
                d2, data_type, n, self.model_name, self.version, suffix,
            )

        if data_type == "custom_generations":
            path = "{}/{}/{}back_{}back_{}_{}_{}.json".format(
                d2, data_type, n, m, i, self.model_name, self.version,
            )

        if data_type == "logprobs":
            suffix = "_curriculum" if self.curriculum else ""
            path = "{}/{}/{}back,{},{},{},{}_{}_{}{}".format(
                d2, data_type, n, demo, i, response_types[0], response_types[1], self.model_name, self.version, suffix,
            )

            if os.path.isfile(f"{path}.json"):
                path = f"{path}.json"
            else:
                path = f"{path}_0_50.json"

        with open(path, "r") as f:
            return json.load(f)

    def _store(
        self,
        data: Union[np.ndarray, dict],
        data_type: str,
        file_name: str,
    ) -> None:
        """
        Stores outputs, which could be numpy arrays, charts or dictionaries.
        """
        path = f"{self.directory}/outputs/{data_type}"
        if not os.path.exists(path):
            os.makedirs(path)
        path = f"{path}/{file_name}"

        if data_type in ["hidden_states", "attentions", "retrieval_attentions"]:
            np.save(path, data)
            return

        if data_type == "attention_maps":
            data.savefig(path)
            data.clear()
            del data
            gc.collect()
            return

        if "charts" in data_type:
            data.savefig(path, dpi=800, bbox_inches='tight')
            return

        with open(path, "w") as f:
            json.dump(data, f)

    def _update(
        self,
        x: str,
        i: int = 0,
        response_types: list[str] = ["gen","gen"],
        past_dialog: list[dict] = None,
    ) -> None:
        """
        Updates current dialog with stimuli (user messages) and responses
        (assistant messages).

        Args:
            x: String of 24 letter stimuli.
            i: Index ranging from 0 to 24.
            response_types: First element specifies response type for x[:i]. If
                `gen`, prompt language model. If `kback`, add correct k-back
                responses. If `past`, add past n-back responses from language
                model. Second element is analogous but for x[i:].
            past_dialog: Past dialog.
        """
        for j in range(0, len(x)):
            self.dialog += [{"role": "user", "content": x[j]}]
            response_type = response_types[i<=j]

            if "back" in response_type:
                m = int(response_type.replace("back", ""))
                cache = [x[j-k-1] if j-k-1 >= 0 else "none" for k in range(-1,m)]
                response = self.prompter("assistant", m, cache=cache)

            if response_type == "gen":
                response = self.client.chat.completions.create(
                    model=f"{self.model_dir}/{self.model_name}",
                    messages=self.dialog,
                ).choices[0].message.content

            if response_type == "past":
                response = past_dialog[len(past_dialog)-49+j*2+1]["content"]

            self.dialog += [{"role": "assistant", "content": response}]

    def _update_system(
        self,
        n: int,
        mode: str,
    ) -> None:
        """
        Adds instructions (system message) to current dialog.
        """
        content = self.prompter("system", n, mode=mode)
        self.dialog += [{"role": "system", "content": content}]

    def generate(
        self,
        n: int,
    ) -> None:
        """
        Recursively prompts model to do n-back task, providing demo.
        """
        if not self.curriculum:
            task_data = self._fetch("task", n)
        else:
            task_data = [self._fetch("task",k) for k in range(1,n+1)]

        dialogs = []
        for i in range(0, 50):
            self.dialog = []

            if not self.curriculum and not self.interactive:
                self._update_system(n, "demo")
                self._update(
                    task_data[i]["x"]["demo"],
                    i=0,
                    response_types=[f"{n}back",f"{n}back"],
                )
                self._update_system(n, "test")
                self._update(task_data[i]["x"]["test"])

            if self.interactive:
                self.dialog.append(
                    self._fetch("interactive_demos", n)[f"{self.model_dir}/{self.model_name}"],
                )
                self._update(task_data[i]["x"]["test"])

            if self.curriculum:
                for m in range(n):
                    self._update_system(m+1, "demo")
                    self._update(
                        task_data[m][i]["x"]["demo"],
                        i=0,
                        response_types=[f"{m+1}back",f"{m+1}back"],
                    )
                self._update_system(n, "test")
                self._update(task_data[-1][i]["x"]["test"])

            dialogs.append(copy.deepcopy(self.dialog))
            print(f"\r {n}back {self.version} {self.model_name} {i+1}/50", end="", flush=True)

        suffix = "_interactive" if self.interactive else "_curriculum" if self.curriculum else ""
        file_name = f"{n}back_{self.model_name}_{self.version}{suffix}.json"
        self._store(dialogs, "generations", file_name)

    def custom_generate(
        self,
        n: int,
        m: int,
        i: int,
    ) -> None:
        """
        Recursively prompts model to do n-back task, providing n-back instructions,
        n-back demo, and m-back completions up to step i.
        """
        task_data = self._fetch("task", n)

        dialogs = []
        for j in range(0, 50):
            self.dialog = []

            self._update_system(n, "demo")
            self._update(
                task_data[j]["x"]["demo"],
                i=0,
                response_types=[f"{n}back",f"{n}back"],
            )
            self._update_system(n, "demo")
            self._update(
                task_data[j]["x"]["test"],
                i=i,
                response_types=[f"{m}back","gen"],
            )

            dialogs += [copy.deepcopy(self.dialog)]
            print(f"\r {n}back {m}back {self.version} {self.model_name} i={i} {j+1}/50", end="", flush=True)

        file_name = f"{n}back_{m}back_{i}_{self.model_name}_{self.version}.json"
        self._store(dialogs, "custom_generations", file_name)

    def _compare(
        self,
        dialogs: list[dict],
        n: int,
        m: int = -1,
    ) -> np.ndarray:
        """
        Compares model generations with correct m-back responses.

        Args:
            dialogs: List of dialogs.
            n: Fetch model generations for n-back task.
            m: If unspecified, compare model generations with correct n-back
                responses. If specified, compare model generations with correct
                m-back responses.
        Returns:
            Binary `np.array` of shape (number of trials, sequence length, number
            of comparisons per response), which is (50, 24, 3) for v1.
        """
        m = n if m == -1 else m
        task_data = self._fetch("task", n)

        ret = []
        for i, dialog in enumerate(dialogs):
            ret.append([])

            self.dialog = []
            self._update(task_data[i]["x"]["test"], i=0, response_types=[f"{m}back",f"{m}back"])

            for message, target in zip(dialog[-48:], self.dialog):
                if message["role"] != "assistant":
                    continue

                ret[-1].append([])

                if self.version == "v1":
                    pattern = r".*(\w+) and (\w+) are (\w+)."

                if self.version == "v2":
                    s = "current: (\w+), " +  ", ".join(f"{i+1} back: (\w+)" for i in range(n)) + \
                        f"; current letter (\w+) and letter {n} back (\w+) are (\w+)."
                    pattern = re.compile(s)

                f = lambda x: re.match(pattern, x).groups()
                a = f(message["content"].lower().replace("  "," "))
                b = f(target["content"])
                for ai, bi in zip(a, b):
                    ret[-1][-1].append(ai == bi)

        return np.array(ret)

    def analyze_generations(
        self,
        n: int,
        graph: bool = True,
    ) -> None:
        """
        Computes accuracy for each component of model response, including
        retrieval and label accuracies, and plots n-back consistent retrieval
        accuracy over time for each m <= n.
        """
        def get_stats(x, a):
            accs = np.mean(x[:,:,a], axis=1)
            acc = np.mean(accs)
            ci = stats.t.interval(
                0.9,
                len(accs)-1,
                loc=acc,
                scale=stats.sem(accs)
            )
            ci = 0 if math.isnan(ci[0]) else ci[1] - ci[0]
            return acc, ci

        dialogs = self._fetch("generations", n)
        x = self._compare(dialogs, n)
        print(f"{n}-back")
        print(self.model_name)
        print("Retrival / Label Accuracy")
        print("%.2f / %.2f" % (
            get_stats(x,x.shape[-1]-2)[0], 
            get_stats(x,x.shape[-1]-1)[0]
        ))

        if self.version == "v1" and graph:
            a = [self._compare(dialogs, n, m=m).sum(axis=0) for m in range(1,n+1)]
            b = np.transpose(np.array(a), (2,0,1))
            fig = plt.figure(figsize=(5,3))
            colors = ["#008080", "#ffb74d", "#4caf50"]

            for m in range(1, b.shape[-2]+1):
                plt.plot(
                    [k for k in range(n+1,25)],
                    b[-2,m-1,n:] / 50,
                    label=f"m = {str(m)}",
                    color=colors[m-1],
                    linewidth=2,
                    marker="o",
                    markeredgewidth=1,
                    markersize=5,
                )

            plt.xlabel("Time step")
            plt.xlim(n, 25)
            plt.xticks([5,10,15,20])
            plt.ylabel("Retrieval accuracy")
            plt.ylim(-0.1, 1.1)
            plt.legend()
            plt.grid(True)
            file_name = f"{n}back_{self.model_name}_{self.version}.png"
            self._store(plt, "generation_charts", file_name)
            plt.show()

    def analyze_custom_generations(
        self,
        n: int,
    ) -> None:
        """
        Plots n-back consistent retrieval accuracy over time for each m <= n.
        """
        fig = plt.figure(figsize=(5,3))
        colors = ["#008080", "#ffb74d", "#4caf50"]

        for m in range(1, n+1):
            a = []
            for i in range(23):
                dialog = self._fetch("custom_generations", n, m=m, i=i)
                a.append(np.mean(self._compare(dialog,n,m=m)[:,i+1:,:], axis=1))
            a = np.mean(np.array(a), axis=1)
            b = np.transpose(a)

            plt.plot(
                [k for k in range(n+1,24)],
                b[-2,n:],
                label=f"m = {str(m)}",
                color=colors[m-1],
                linewidth=2,
                marker="o",
                markeredgewidth=1,
                markersize=5,
            )

        plt.xlabel("Time step")
        plt.xlim(n, 24)
        plt.xticks([5,10,15,20])
        plt.ylabel("Retrieval accuracy")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.grid(True)
        file_name = f"{n}back_mback_{self.model_name}_{self.version}.png"
        self._store(plt, "custom_generation_charts", file_name)
        plt.show()

    def get_logprobs(
        self,
        n: int,
        demo: int = 1,
        i: int = 0,
        response_types: list[str] = ["past","past"],
    ) -> None:
        """
        Gets token logprobs for `self.dialog`.

        Args:
            n: Use n-back task instructions.
            demo: If 1, include n-back task demo. If 0, exclude demo.
            i: Index ranging from 0 to 24.
            response_types: See `self._update()`.
        """
        rsp = ",".join(response_types)

        if not self.curriculum:
            task_data = self._fetch("task", n)
        else:
            task_data = [self._fetch("task",k) for k in range(1,n+1)]

        dialogs = self._fetch("generations", n) if "past" in response_types else [None] * 50
        logprobs = []
        for j in range(0, 50):
            self.dialog = []

            if not self.curriculum:
                self._update_system(n, "demo")
                if demo:
                    self._update(
                        task_data[j]["x"]["demo"],
                        i=0,
                        response_types=[f"{n}back",f"{n}back"],
                    )
                    self._update_system(n, "test")
                self._update(
                    task_data[j]["x"]["test"],
                    i=i,
                    response_types=response_types,
                    past_dialog=dialogs[j],
                )

            if self.curriculum:
                for m in range(n):
                    self._update_system(m+1, "demo")
                    self._update(
                        task_data[m][j]["x"]["demo"],
                        i=0,
                        response_types=[f"{m+1}back",f"{m+1}back"],
                    )
                self._update_system(n, "test")
                self._update(
                    task_data[-1][j]["x"]["test"],
                    i=0,
                    response_types=response_types,
                )

            response = self.client.chat.completions.create(
                model=f"{self.model_dir}/{self.model_name}",
                messages=self.dialog,
                logprobs=1,
                echo=1,
                max_tokens=1,
            )

            logprobs.append(
                {
                    "token_logprobs": response.prompt[0].logprobs.token_logprobs,
                    "tokens": response.prompt[0].logprobs.tokens,
                    "token_ids": response.prompt[0].logprobs.token_ids,
                }
            )

            print(f"\r {n}back demo={demo} i={i} rsp=[{rsp}] {self.version} {self.model_name} {j+1}/50", end="", flush=True)

        suffix = "_curriculum" if self.curriculum else ""
        file_name = "{}back,{},{},{}_{}_{}{}.json".format(
            n, demo, i, rsp, self.model_name, self.version, suffix,
        )
        self._store(logprobs, "logprobs", file_name)

    def _get_retrieval_indices(
        self,
        tokens: list[str],
        skip_demo: bool,
        forward: bool = False,
        i: int = 0,
    ) -> list[int]:
        """
        Gets indices of retrieval tokens.

        Args:
            tokens: List of tokens.
            skip_demo: If `True`, skip demo.
            forward: If `True`, start from the first token. If `False`, start from
                the last token.
            i: Index ranging from 0 to 24. Number of retrieval tokens to skip.

        Returns:
            List of indices of retrieval tokens.
        """
        if not forward:
            def _previous(index, role):
                index -= 1
                while tokens[index] != role:
                    index -= 1
                return index

            ret = []
            j = len(tokens) - 1
            for _ in range(24):
                j = _previous(j, "assistant")
                k = j
                while not ("and" in tokens[k-1] and "are" in tokens[k+1]):
                    k += 1
                ret.insert(0, k)

        if forward:
            if self.model_name == "gemma-2-27b-it":
                count = 0
                for n, token in enumerate(tokens):
                    if token == "undefined":
                        if count == 0 or count == 49:
                            tokens[n] = "system"
                        elif count < 49:
                            tokens[n] = "user" if count % 2 else "model"
                        else:
                            tokens[n] = "model" if count % 2 else "user"
                        count += 1

            def _next(index, role):
                index += 1
                while tokens[index+1] != role:
                    index += 1
                return index

            j = 0
            while tokens[j+1] != "system":
                j += 1

            if skip_demo:
                j = _next(j, "system")

            j = _next(j, "model" if "gemma" in self.model_name else "assistant")
            for _ in range(i):
                j = _next(j, "model" if "gemma" in self.model_name else "assistant")

            ret = []
            while j < len(tokens) - 1:
                if "and" in tokens[j-1] and "are" in tokens[j+1]:
                    ret.append(j)
                j += 1

        return ret


    def _get_source_indices(
        self,
        tokens: list[str],
    ) -> list[int]:
        """
        Gets indices of retrival tokens, including demo.

        Args:
            tokens: List of tokens.

        Returns:
            List of indices of retrival tokens.
        """
        ret = []
        for i in range(3, len(tokens)-1):
            if "Qwen" in self.model_name and tokens[i-3] == "<|im_start|>" and \
                tokens[i-2] == "user" and tokens[i+1] == "<|im_end|>":
                ret += [i]

            if "Llama" in self.model_name and "user" in tokens[i-3]:
                ret += [i]

        return ret

    def analyze_logprobs(
        self,
        n: int,
        m_range: list[int] = [1,2,3],
        strip: bool = True,
        line: bool = True,
    ) -> None:
        """
        Visualizes retrieval logprobs.

        Args:
            n: n-back task.
            m_range: m-back-consistent continuations to analyze.
            strip: If `True`, generate strip plots.
            line: If `True`, generate line plots.
        """
        colors = ["#008080", "#ffb74d", "#4caf50"]
        sns.set_palette(colors)

        if strip and len(m_range) == 3:
            fig, axes = plt.subplots(1, 2, figsize=(5,3.5), sharey=True)

            x, y = [], []
            for m in m_range:
                logprobs = self._fetch("logprobs", n, demo=0, i=0, response_types=[f"{m}back",f"{m}back"])
                for trial_logprobs in logprobs:
                    indices = self._get_retrieval_indices(trial_logprobs["tokens"], 0, forward=True)
                    token_logprobs = np.array(trial_logprobs["token_logprobs"])
                    x += [f"{m}-back"]
                    y += [token_logprobs[indices].mean()]

            sns.stripplot(ax=axes[0], x=x, y=y, hue=x, edgecolor="black", linewidth=0.5)
            axes[0].set_title("Without demo", fontsize=10)
            axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0].grid(True)

            x, y = [], []
            for m in m_range:
                logprobs = self._fetch("logprobs", n, demo=1, i=0, response_types=[f"past", f"{m}back"])
                for trial_logprobs in logprobs:
                    indices = self._get_retrieval_indices(trial_logprobs["tokens"], 1, forward=True)
                    token_logprobs = np.array(trial_logprobs["token_logprobs"])
                    x += [f"{m}-back"]
                    y += [token_logprobs[indices].mean()]

            sns.stripplot(ax=axes[1], x=x, y=y, hue=x, edgecolor="black", linewidth=0.5)
            axes[1].set_title("With demo", fontsize=10)
            axes[1].set_ylim(-6, 0.2)
            axes[1].grid(True)

            fig.text(0, 0.5, "Retrieval logprob", ha="center", va="center", rotation="vertical")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            file_name = f"{n}back_{self.model_name}_{self.version}_strip.png"
            self._store(fig, "logprob_charts", file_name)
            plt.show()

        if strip and len(m_range) == 10:
            fig, ax = plt.subplots(figsize=(6,2.5))

            x, y = [], []
            for m in m_range:
                logprobs = self._fetch("logprobs", n, demo=1, i=0, response_types=[f"{m}back",f"{m}back"])
                for trial_logprobs in logprobs:
                    indices = self._get_retrieval_indices(trial_logprobs["tokens"], 1)
                    token_logprobs = np.array(trial_logprobs["token_logprobs"])
                    x += [f"{m}"]
                    y += [token_logprobs[indices].mean()]

            sns.stripplot(ax=ax, x=x, y=y, hue=x, edgecolor="black", linewidth=0.5)
            ax.tick_params(axis='y', which='both', length=0, labelsize=0)
            plt.grid(True)
            plt.tight_layout(rect=[0,0,1,0.95])
            suffix = "_curriculum" if self.curriculum else ""
            file_name = f"{n}back_{self.model_name}_{self.version}_strip_10{suffix}.png"
            self._store(fig, "logprob_charts", file_name)
            plt.show()

        if line:
            x, y = [i for i in range(n+1,25)], []
            for m in range(1, n+1):
                y.append([])
                for i in range(24):
                    y[-1].append([])
                    for logprobs in self._fetch("logprobs", n, demo=1, i=i, response_types=["past",f"{m}back"]):
                        indices = self._get_retrieval_indices(logprobs["tokens"], 1, forward=True, i=i)
                        y[-1][-1].append(logprobs["token_logprobs"][indices[0]])
            y = np.array(y).mean(axis=-1)

            fig, ax = plt.subplots(figsize=(5,3))

            for m in range(1, n+1):
                plt.plot(
                    [k for k in range(n+1,25)],
                    y[m-1][n:],
                    label=f"m = {str(m)}",
                    color=colors[m-1],
                    linewidth=2,
                    marker="o",
                    markeredgewidth=1,
                    markersize=5
                )

            ax.yaxis.set_label_coords(-0.1, 0.5)
            plt.xlabel("Time step")
            plt.xlim(n, 25)
            plt.ylabel("Retrieval logprob")
            plt.ylim(min(-10,np.min(y[:,n:])-1), 1)
            plt.xticks([5,10,15,20])
            plt.legend()
            plt.grid(True)
            file_name = f"{n}back_{self.model_name}_{self.version}.png"
            self._store(fig, "logprob_charts", file_name)
            plt.show()

    def get_average_entropy(
        self,
        n: int,
        m_range: list[int] = [1,2,3,4,5,6,7,8,9,10],
    ) -> int:
        """
        Computes average entropy of retrieval token logprobs over `m_range`.
        """
        retrieval_lps = [[] for _ in range(50)]
        for m in m_range:
            logprobs = self._fetch("logprobs", n, demo=1, i=0, response_types=[f"{m}back",f"{m}back"])
            for trial, trial_logprobs in enumerate(logprobs):
                indices = self._get_retrieval_indices(trial_logprobs["tokens"], 1)
                token_logprobs = np.array(trial_logprobs["token_logprobs"])
                retrieval_lps[trial] += [token_logprobs[indices].mean()]

        def entropy(lp):
            p = np.exp(np.array(lp))
            p /= np.sum(p)
            return stats.entropy(p)

        return np.mean([entropy(x) for x in retrieval_lps])

    def get_hidden_states(
        self,
        n: int,
    ) -> None:
        """
        Gets hidden states for each trial.
        """
        for trial, dialog in enumerate(self._fetch("generations",n)):
            input_ids = self.tokenizer.encode(
                self.tokenizer.apply_chat_template(dialog, tokenize=False),
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = torch.stack(outputs.hidden_states).numpy()

            file_name = f"{n}back_{self.model_name}_{self.version}_#{trial}.npy"
            self._store(hidden_states, "hidden_states", file_name)
            print(f"\r {n}back {self.model_name} {trial+1}/50", end="", flush=True)

    def get_attentions(
        self,
        n: int,
    ) -> None:
        """
        Gets attentions for each trial.
        """
        for trial, dialog in enumerate(self._fetch("generations",n)):
            input_ids = self.tokenizer.encode(
                self.tokenizer.apply_chat_template(dialog, tokenize=False),
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=True)
            attentions = torch.stack(outputs.attentions).numpy()[:,0,:,:,:]

            file_name = f"{n}back_{self.model_name}_{self.version}_#{trial}.npy"
            self._store(attentions, "attentions", file_name)
            print(f"\r {n}back {self.model_name} {trial+1}/50", end="", flush=True)

    def get_mean_retrieval_attentions(
        self,
        n: int,
    ) -> None:
        """
        Computes mean retrieval attention for each (trial, head, layer) and plots
        attention with the highest value.
        """
        attention_filter = np.ones((96,96))
        for k in range(1, 25-n):
            a = 2 * (n + k) - 1
            b = a + 48
            c = 2 * n + 1
            attention_filter[a][a-c] = 0
            attention_filter[b][b-c] = 0

        max_value, max_attention, max_tokens = 0, None, None
        max_trial, max_layer, max_head = None, None, None
        mean_retrieval_attentions = []
        for i, dialog in enumerate(self._fetch("generations",n)):
            mean_retrieval_attentions.append([[]])

            input_ids = self.tokenizer.encode(
                self.tokenizer.apply_chat_template(dialog, tokenize=False),
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=True)
            attentions = [tensor.cpu() for tensor in outputs.attentions]
            attentions = torch.stack(attentions).numpy()[:,0,:,:,:]

            tokens = self.tokenizer.tokenize(
                self.tokenizer.apply_chat_template(dialog, tokenize=False),
            )

            indices = self._get_retrieval_indices(tokens, False, forward=True) + \
                      self._get_source_indices(tokens)
            indices.sort()
            tokens = [tokens[i] for i in indices]

            num_layers = attentions.shape[0]
            num_heads = attentions.shape[1]

            mean_retrieval_attentions_trial = [
                [None for _ in range(num_heads)] for _ in range(num_layers)
            ]

            for layer in range(num_layers):
                for head in range(num_heads):
                    attention = attentions[layer][head]
                    attention = attention[indices,:]
                    attention = attention[:,indices]

                    diff = attention - attention_filter
                    diff = diff[diff > 0]
                    mean_retrieval_attention = sum(diff) / (48-2*n)

                    if mean_retrieval_attention > max_value:
                        max_value, max_attention, max_tokens = mean_retrieval_attention, attention, tokens
                        max_trial, max_layer, max_head = i+1, layer, head

                    mean_retrieval_attentions_trial[layer][head] = mean_retrieval_attention

                    print(f"\r {n}back {self.model_name} {i+1}/50 {layer}.{head} {mean_retrieval_attention}", end="", flush=True)

            mean_retrieval_attentions.append(mean_retrieval_attentions_trial)

        file_name = f"{n}back_{self.model_name}_{self.version}.npy"
        self._store(mean_retrieval_attentions, "mean_retrieval_attentions", file_name)

        plt.ioff()
        fig = plt.figure(figsize=(20,20), dpi=300)
        sns.heatmap(
            max_attention,
            xticklabels=max_tokens,
            yticklabels=max_tokens,
            cmap="Greens",
            cbar=False,
        )
        plt.xticks(rotation=90)
        file_name = f"{n}back_{self.model_name}_{self.version}_#{max_trial}_{layer}.{head}_mra={max_value}.png"
        self._store(fig, "attention_maps", file_name)
        fig.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key_together")
    parser.add_argument("--api_key_openai")
    parser.add_argument("--use_auth_token")
    args = parser.parse_args()
    k1 = args.api_key_together
    k2 = args.api_key_openai
    use_auth_token = args.use_auth_token
    directory = os.getcwd()

    model_names = [
        "gpt-3.5-turbo",
        "Qwen/Qwen1.5-14B-Chat",
        "Qwen/Qwen1.5-32B-Chat",
        "Qwen/Qwen2-72B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    ]

    # sec 4.1
    # sec 4.2
    # sec 4.3
    
    for x in range(8):
        for n in [1, 2, 3]:
            api_key = k2 if "gpt" in model_names[x] else k1
            t = NBackTest(directory, "v1", model_names[x], use_api=True, api_key=api_key)
            t.generate(n)

    for x in range(8):
        for n in [1, 2, 3]:
            t = NBackTest(directory, "v1", model_names[x])
            t.analyze_generations(n)

    for x in range(1, 8):
        for n in [1, 2, 3]:
            for m in [1, 2, 3]:
                t = NBackTest(directory, "v1", model_names[x], use_api=True, api_key=k1)
                t.get_logprobs(n, demo=0, i=0, response_types=[f"{m}back",f"{m}back"])

    for x in range(1, 8):
        for n in [1, 2, 3]:
            for m in [1, 2, 3]:
                for i in range(25):
                    t = NBackTest(directory, "v1", model_names[x], api_key=k1)
                    t.get_logprobs(n, demo=1, i=i, response_types=["past",f"{m}back"], use_api=True, api_key=k1)

    for x in range(1, 8):
        for n in [1, 2, 3]:
            t = NBackTest(directory, "v1", model_names[x])
            t.analyze_logprobs(n)

    for n in [2, 3]:
        for m in [1, 2, 3]:
            for i in range(25):
                t = NBackTest(directory, "v1", model_names[-1], use_api=True, api_key=k1)
                t.custom_generate(n, m, i)

    for m in [2, 3]:
        t = NBackTest(directory, "v1", model_names[-1])
        t.analyze_custom_generations(m)
    
    # sec 4.4

    for n in range(4, 11):
        t = NBackTest(directory, "v1", model_names[-3], use_api=True, api_key=k1)
        t.generate(n)

    for n in range(4, 11):
        t = NBackTest(directory, "v1", model_names[-3])
        t.analyze_generations(n, graph=False)

    for n in range(1, 11):
        for m in range(1, 11):
            t = NBackTest(directory, "v1", model_names[-3], use_api=True, api_key=k1)
            t.get_logprobs(n, demo=1, i=0, response_types=[f"{m}back",f"{m}back"])

    for n in range(1, 11):
        t = NBackTest(directory, "v1", model_names[-3])
        t.analyze_logprobs(n, m_range=[1,2,3,4,5,6,7,8,9,10], strip=True, line=False)

    # sec 4.5

    for n in range(1, 11):
        t = NBackTest(directory, "v1", model_names[-3], use_api=True, api_key=k1, curriculum=True)
        t.generate(n)

    for n in range(1, 11):
        t = NBackTest(directory, "v1", model_names[-3], use_api=True, api_key=k1, curriculum=True)
        t.analyze_generations(n, graph=False)

    for n in range(1, 11):
        for m in range(1, 11):
            t = NBackTest(directory, "v1", model_names[-3], use_api=True, api_key=k1, curriculum=True)
            t.get_logprobs(n, demo=1, i=0, response_types=[f"{m}back",f"{m}back"])

    for n in range(1, 11):
        t = NBackTest(directory, "v1", model_names[-3], curriculum=True)
        t.analyze_logprobs(n, m_range=[1,2,3,4,5,6,7,8,9,10], strip=True, line=False)

    # sec 4.6

    for x in [-1, -3]:
        for n in [2, 3]:
            t = NBackTest(directory, "v1", model_names[x], use_api=True, api_key=k1, interactive=True)
            t.generate(n)
    
    for x in [-1, -3]:
        for n in [2, 3]:
            t = NBackTest(directory, "v1", model_names[x], use_api=True, api_key=k1, interactive=True)
            t.analyze_generations(n)

    # sec 4.7

    for x in range(8):
        for n in [1, 2, 3]:
            api_key = k2 if "gpt" in model_names[x] else k1
            t = NBackTest(directory, "v2", model_names[x], use_api=True, api_key=api_key)
            t.generate(n)

    for x in range(8):
        for n in [1, 2, 3]:
            t = NBackTest(directory, "v2", model_names[x])
            t.analyze_generations(n)

    # sec 4.8

    for x in [3, 5]:
        t = NBackTest(directory, "v1", model_names[x], use_model=True, use_tokenizer=True, use_auth_token=use_auth_token)
        t.get_mean_retrieval_attentions(2)


if __name__ == "__main__":
    main()
