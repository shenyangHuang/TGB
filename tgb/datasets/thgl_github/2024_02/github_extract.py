import json
from datetime import datetime
import glob
import gzip
import csv
"""
go to https://www.gharchive.org/

wget https://data.gharchive.org/2024-01-{01..31}-{0..23}.json.gz

Creates (src, edge_type, dst, time) edges from the GitHub archive JSON file.
Using the rules from https://arxiv.org/pdf/2007.01231 (page 11)
The parser creates 18 rules that are in the GITHUB-SE-1Y-Repo dataset. I wrote the meaning of the rules and sources and destination types here.
"""



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
    try:
        if "action" not in event["payload"]:
            return []
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
    except:
        return []


def parse_issue_event(event):
    try:
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
    except:
        return []


def parse_pull_request_event(event):
    try:
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
    except:
        return []

def parse_pull_request_review_comment_event(event):
    try:
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
    except:
        return []


def parse_fork_event(event):
    try:
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
    except:
        return []


def parse_member_event(event):
    try:
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
    except:
        return []


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
    # events = []
    output_dict = {}
    num_edge = 1
    #with open(filename) as f:
    with gzip.open(filename, 'r') as f:
        for i, line in enumerate(f):
            djson = json.loads(line)
            parsed_events = parse_event(djson)
            if (len(parsed_events) > 0):
                for edge in parsed_events:
                    #? ['/user/41898282', 'U_SE_O_I', '/issue/2061196208', 1704085558]
                    ts = int(edge[3])
                    head = edge[0]
                    rel = edge[1]
                    tail = edge[2]
                    if ts not in output_dict:
                        output_dict[ts] = {}
                        output_dict[ts][(head,tail,rel)] = 1
                        num_edge += 1
                    else:
                        if (head,tail,rel) in output_dict[ts]:
                            output_dict[ts][(head,tail,rel)] += 1
                        else:
                            output_dict[ts][(head,tail,rel)] = 1
                            num_edge += 1
    print("Parsed {} events".format(num_edge))
    return output_dict

def write2csv(outname, out_dict):
    with open(outname, 'a') as f:
        writer = csv.writer(f, delimiter =',')
        # writer.writerow(['ts', 'head', 'tail', 'relation_type'])
        ts_list = list(out_dict.keys())
        ts_list.sort()

        for ts in ts_list:
            for edge in out_dict[ts]:
                head = edge[0]
                tail = edge[1]
                relation_type = edge[2]
                row = [ts, head, tail, relation_type]
                writer.writerow(row)



def main():
    total_edge_dict = {}
    for file in glob.glob("*.json.gz"):
        print ("processing,", file)
        edge_dict = parse_file(file)
        # print ('check for edge overlap')
        # print(edge_dict.keys() & total_edge_dict.keys())
        # print ("-------------------------")
        #! write to csv after each file is processed. 
        # total_edge_dict.update(edge_dict)
        outname = "github_02_2024.csv"
        write2csv(outname, edge_dict)
        






if __name__ == "__main__":
    main()