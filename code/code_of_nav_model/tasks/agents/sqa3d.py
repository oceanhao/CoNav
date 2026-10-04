from .llava import LLaVAAgent


class SQA3DAgent(LLaVAAgent):
    name = "sqa3d"

    def get_prompt(self, task, *args, **kwargs):
        if task == "3dqa":
            return self.get_3dqa_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_3dqa_prompt(self, ques, cand_num):
        obs_text = " ".join(["({}) <cand>".format(i) for i in range(cand_num)])
        prompt = (
            "Please answer questions based on the observation.\n"
            + "The following is the Observation, which includes multiple images from different locations.\n"
            + "### Observation: {} \n".format(obs_text)
            + "### Question: {}\n".format(ques)
            + "Following is the 3D Spatial Description of the current location, only for reference.\n"
            + "### Spatial Description: <3d_f>\n"
            + "### Your Answer: "
        )
        return prompt
