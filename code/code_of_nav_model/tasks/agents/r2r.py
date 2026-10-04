from .mp3d_agent import MP3DAgent
import os


class R2RAgent(MP3DAgent):
    name = "r2r"

    def get_prompt(self, task, *args, **kwargs):
        if task == "navigation":
            return self.get_navigation_prompt(*args, **kwargs)
        elif task == "summarization":
            return self.get_summarization_prompt(*args, **kwargs)
        elif task == "embodied_qa":
            return self.get_embodied_qa_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_navigation_prompt(self, instruction, hist_num, cand_num, cls_token):

        prompt = "### Instruction: Navigate following the instruction. {} \n".format(
            instruction
        )

        prompt += "Following is the History, which contains the visual information of your previous decisions.\n"
        hist_text = " ".join(["({}) <hist>".format(i) for i in range(hist_num)])
        prompt += "### History: {}\n".format(hist_text)

        prompt += "Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n"
        obs_text = " ".join(
            ["({}) <cand>".format(i) if i > 0 else "(0) stop" for i in range(cand_num)]
        )
        prompt += "### Candidate: {}\n".format(obs_text)
        if os.environ.get("use_angle") == "use":

            prompt += (
                "Following is the Candidate Angle description, angle (0) is stop.\n"
            )
            angle_text = " ".join(
                [
                    "({}) <angle>".format(i) if i > 0 else "(0) stop"
                    for i in range(cand_num)
                ]
            )
            prompt += "### Candidate Angle: {}\n".format(angle_text)

        prompt += "Following is the 3D Spatial Description of the current location, only for reference.\n"
        prompt += "### Spatial Description: <3d_f>\n"

        prompt += "Compare the History and Instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location.\n"
        prompt += "### Output: {}".format(cls_token)

        return prompt

    def get_summarization_prompt(self, instruction, hist_num, cand_num):

        prompt = f"### Instruction: Predict the fine-grained instruction based on your previous history and current location. Fine-grained instructions contain commands for each individual step. \n"

        prompt += "Following is the History, which contains the visual information of your previous decisions.\n"
        hist_text = " ".join(["({}) <hist>".format(i) for i in range(hist_num)])
        prompt += "### History: {}\n".format(hist_text)

        if cand_num != 0:
            prompt += "Following is the Observation, which contains panoramic views at your current location.\n"
            obs_text = " ".join(["({}) <cand>".format(i) for i in range(cand_num)])
            prompt += "### Candidate: {}\n".format(obs_text)

        prompt += "Following is the 3D Spatial Description of the current location, only for reference. \n"
        prompt += "### Spatial Description: <3d_f>\n"

        prompt += "Please generate the step-by-step instruction.\n"
        prompt += "### Answer: "

        return prompt

    def get_embodied_qa_prompt(self, instruction, hist_num, cand_num):

        prompt = f"### Instruction: answer the question. \n"

        if hist_num != 0:
            prompt += "Following is the History, which contains the visual information of your previous decisions.\n"
            hist_text = " ".join(["({}) <hist>".format(i) for i in range(hist_num)])
            prompt += "### History: {}\n".format(hist_text)

        if cand_num != 0:
            prompt += "Following is the Observation, which contains panoramic views at your current location.\n"
            obs_text = " ".join(["({}) <cand>".format(i) for i in range(cand_num)])
            prompt += "### Candidate: {}\n".format(obs_text)

        prompt += "### Question: {}\n".format(instruction)
        prompt += "### Answer: "

        return prompt


class R2RAugAgent(R2RAgent):
    name = "r2r_aug"
