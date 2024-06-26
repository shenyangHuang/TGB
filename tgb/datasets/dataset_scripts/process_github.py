import json
from datetime import datetime

rels = {
    "IC_Created_IC_I": "IC_AO_C_I",
    "IC_Created_U_IC": "U_SO_C_IC",
    "I_Opened_U_I": "U_SE_O_I",
    "I_Opened_I_R": "I_AO_O_R",
    "I_Closed_U_I": "U_SE_C_I",
    "I_Closed_I_R": "I_AO_C_R",
    "I_Reopened_U_I": "U_SE_RO_I",
    "I_Reopened_I_R": "I_AO_RO_R",
    "PR_Opened_U_PR": "U_SO_O_P",
    "PR_Opened_PR_R": "P_AO_O_R",
    "PR_Closed_U_PR": "U_SO_C_P",
    "PR_Closed_PR_R": "P_AO_C_R",
    "PR_Reopened_U_PR": "U_SO_R_P",
    "PR_Reopened_PR_R": "P_AO_R_R",
    "PRRC_Created_U_PRC": "U_SO_C_PRC",
    "PRRC_Created_PRC_PR": "PRC_AO_C_P",
    "Forked_R_R": "R_FO_R",
    "AddMember_U_R": "U_CO_A_R",
}

issue_comment_format = "/issue_comment/{}"
issue_format = "/issue/{}"
user_format = "/user/{}"
repo_format = "/repo/{}"
pull_request_format = "/pr/{}"
pull_request_review_comment_format = "/pr_review_comment/{}"


def str_to_timestamp(time_str):
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(dt.timestamp())


def parse_issue_comment_events(event):
    if event["payload"]["action"] == "created":
        issue_comment_id = event["payload"]["comment"]["id"]
        issue_id = event["payload"]["issue"]["id"]
        user_id = event["actor"]["id"]
        created_at = str_to_timestamp(event["created_at"])

        ici_event = [
            issue_comment_format.format(issue_comment_id),
            rels["IC_Created_IC_I"],
            issue_format.format(issue_id),
            created_at,
        ]
        uic_event = [
            user_format.format(user_id),
            rels["IC_Created_U_IC"],
            issue_comment_format.format(issue_comment_id),
            created_at,
        ]
        return [ici_event, uic_event]
    return []


def parse_issue_event(event):
    issue_id = event["payload"]["issue"]["id"]
    user_id = event["actor"]["id"]
    repo_id = event["repo"]["id"]
    created_at = str_to_timestamp(event["created_at"])
    action_map = {
        "opened": ("I_Opened_U_I", "I_Opened_I_R"),
        "closed": ("I_Closed_U_I", "I_Closed_I_R"),
        "reopened": ("I_Reopened_U_I", "I_Reopened_I_R"),
    }
    for action, event_rels in action_map.items():
        if event["payload"]["action"] == action:
            ui_event = [
                user_format.format(user_id),
                rels[event_rels[0]],
                issue_format.format(issue_id),
                created_at,
            ]

            ir_event = [
                issue_format.format(issue_id),
                rels[event_rels[1]],
                repo_format.format(repo_id),
                created_at,
            ]
            return [ui_event, ir_event]
    return []


def parse_pull_request_event(event):
    pull_request_id = event["payload"]["pull_request"]["id"]
    user_id = event["actor"]["id"]
    repo_id = event["repo"]["id"]
    created_at = str_to_timestamp(event["created_at"])
    action_map = {
        "opened": ("PR_Opened_U_PR", "PR_Opened_PR_R"),
        "closed": ("PR_Closed_U_PR", "PR_Closed_PR_R"),
        "reopened": ("PR_Reopened_U_PR", "PR_Reopened_PR_R"),
    }
    for action, event_rels in action_map.items():
        if event["payload"]["action"] == action:
            upr_event = [
                user_format.format(user_id),
                rels[event_rels[0]],
                pull_request_format.format(pull_request_id),
                created_at,
            ]

            prr_event = [
                pull_request_format.format(pull_request_id),
                rels[event_rels[1]],
                repo_format.format(repo_id),
                created_at,
            ]
            return [upr_event, prr_event]
    return []


def parse_pull_request_review_comment_event(event):
    pull_request_review_comment_id = event["payload"]["comment"]["id"]
    pull_request_id = event["payload"]["pull_request"]["id"]
    user_id = event["actor"]["id"]
    created_at = str_to_timestamp(event["created_at"])
    if event["payload"]["action"] == "created":
        uprc_event = [
            user_format.format(user_id),
            rels["PRRC_Created_U_PRC"],
            pull_request_review_comment_format.format(pull_request_review_comment_id),
            created_at,
        ]

        prcpr_event = [
            pull_request_review_comment_format.format(pull_request_review_comment_id),
            rels["PRRC_Created_PRC_PR"],
            pull_request_format.format(pull_request_id),
            created_at,
        ]
        return [uprc_event, prcpr_event]
    return []


def parse_fork_event(event):
    forkee_repo_id = event["payload"]["forkee"]["id"]
    forked_repo_id = event["repo"]["id"]
    created_at = str_to_timestamp(event["created_at"])
    return [
        [
            repo_format.format(forkee_repo_id),
            rels["Forked_R_R"],
            repo_format.format(forked_repo_id),
            created_at,
        ]
    ]


def parse_member_event(event):
    user_id = event["payload"]["member"]["id"]
    repo_id = event["repo"]["id"]
    created_at = str_to_timestamp(event["created_at"])
    return [
        [
            user_format.format(user_id),
            rels["AddMember_U_R"],
            repo_format.format(repo_id),
            created_at,
        ]
    ]


event_handler_dict = {
    "IssueCommentEvent": parse_issue_comment_events,
    "IssuesEvent": parse_issue_event,
    "PullRequestEvent": parse_pull_request_event,
    "PullRequestReviewCommentEvent": parse_pull_request_review_comment_event,
    "ForkEvent": parse_fork_event,
    "MemberEvent": parse_member_event,
}


def parse_event(event):
    event_type = event["type"]
    if event_type in event_handler_dict:
        output_list = event_handler_dict[event_type](event)
        # print("Got {} outputs for event type {}".format(len(output_list), event_type))
    else:
        # print("Unknown event type: {}".format(event_type))
        output_list = []
    return output_list


def parse_file(filename):
    events = []
    with open(filename) as f:
        for i, line in enumerate(f):
            event = json.loads(line)
            parsed_events = parse_event(event)
            events.append(parsed_events)
    events = [event for sublist in events for event in sublist]
    print("Parsed {} events".format(len(events)))
    return events


filename = "2015-01-01-15.json"
parse_file(filename)
