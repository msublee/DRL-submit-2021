import yaml
import os.path as osp

from knockknock import slack_sender


def get_alarm(alarm=False, channel=None):
    deco = no_alarm
    if alarm:
        if osp.exists("gpu_alarm.yaml"):
            with open("gpu_alarm.yaml") as f:
                configs = yaml.load(f, Loader=yaml.FullLoader)
                webhook_url = configs["webhook_url"]
                user_mentions = configs["user_mentions"]
            deco = slack_sender(webhook_url=webhook_url, channel=channel, user_mentions=user_mentions)
    return deco


def no_alarm(func):
    def inner(args):
        func(args)
    return inner