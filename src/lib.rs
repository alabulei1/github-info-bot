use dotenv::dotenv;
use flowsnet_platform_sdk::logger;
use github_flows::octocrab::models::issues::{Comment, Issue};
use http_req::{request::Method, request::Request, uri::Uri};
use log;
use openai_flows::{
    chat::{ChatModel, ChatOptions},
    OpenAIFlows,
};
use serde::{Deserialize, Serialize};
use slack_flows::{listen_to_channel, send_message_to_channel, SlackMessage};
use std::env;
use urlencoding;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn run() {
    logger::init();
    dotenv().ok();

    let slack_workspace = env::var("slack_workspace").unwrap_or("secondstate".to_string());
    let slack_channel = env::var("slack_channel").unwrap_or("github-status".to_string());

    listen_to_channel(&slack_workspace, &slack_channel, |sm| {
        handler(&slack_workspace, &slack_channel, sm)
    })
    .await;
}

async fn handler(workspace: &str, channel: &str, sm: SlackMessage) {
    let trigger_word = env::var("trigger_word").unwrap_or("bot@get".to_string());

    let parts: Vec<&str> = sm
        .text
        .split(&trigger_word)
        .nth(1) // skip the part before "bot@get"
        .unwrap_or("") // if "bot@get" is not found, use an empty string
        .split_whitespace()
        .collect();

    let (owner, repo, user_name) = match parts.as_slice() {
        [owner, repo, user, ..] => (owner, repo, user),
        _ => panic!("Input should contain 'bot@get <github_owner> <github_repo> <user_name>'"),
    };

    if sm.text.contains(&trigger_word) {
        if let Some(res) = analyze_commits(owner, repo, user_name).await {
            send_message_to_channel("secondstate", "test-flow", res.clone()).await;
            let commits_summaries = res;
            if let Some(issues) = get_issues(owner, repo, user_name).await {
                let mut issues_summaries = String::new();
                for issue in issues {
                    if let Some(body) = analyze_issue(owner, repo, user_name, issue).await {
                        issues_summaries.push_str(&body);
                        issues_summaries.push_str("\n");
                    }
                }
                send_message_to_channel("ik8", "ch_mid", issues_summaries.clone()).await;

                if let Some(report) =
                    correlate_commits_issues(&commits_summaries, &issues_summaries).await
                {
                    send_message_to_channel(workspace, channel, report).await;
                }
            }
        }
    }
}
#[derive(Debug, Deserialize)]
struct Page<T> {
    pub items: Vec<T>,
    pub total_count: Option<u64>,
}
pub async fn get_issues(owner: &str, repo: &str, user: &str) -> Option<Vec<Issue>> {
    let query = format!("repo:{owner}/{repo} involves:{user}");
    let encoded_query = urlencoding::encode(&query);

    let mut out: Vec<Issue> = vec![];
    let mut total_pages = None;
    for page in 1..=3 {
        if page > total_pages.unwrap_or(3) {
            break;
        }

        let url_str = format!(
            "https://api.github.com/search/issues?q={encoded_query}&sort=created&order=desc&page={page}"
        );

        match github_http_fetch_tokenless(&url_str).await {
            Some(res) => match serde_json::from_slice::<Page<Issue>>(&res) {
                Err(_e) => log::error!("Error parsing Page<Issue>: {:?}", _e),

                Ok(issue_page) => {
                    if total_pages.is_none() {
                        if let Some(count) = issue_page.total_count {
                            total_pages = Some((count / 30) as usize + 1);
                        }
                    }
                    for issue in issue_page.items {
                        out.push(issue);
                    }
                }
            },

            None => {}
        }
    }

    Some(out)
}

pub async fn analyze_issue(owner: &str, repo: &str, user: &str, issue: Issue) -> Option<String> {
    let issue_creator_name = issue.user.login;
    let issue_number = issue.number;
    let issue_title = issue.title;
    let issue_body = match issue.body {
        Some(body) => squeeze_fit_comment_texts(&body, "```", 500, 0.6),
        None => "".to_string(),
    };
    let issue_date = issue.created_at.date_naive().to_string();
    let html_url = issue.html_url.to_string();

    let labels = issue
        .labels
        .into_iter()
        .map(|lab| lab.name)
        .collect::<Vec<String>>()
        .join(", ");

    let mut all_text_from_issue = format!("User '{issue_creator_name}', has submitted an issue titled '{issue_title}', labeled as '{labels}', with the following post: '{issue_body}'.");

    let url_str = format!(
        "https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page=100",
    );

    match github_http_fetch_tokenless(&url_str).await {
        Some(res) => match serde_json::from_slice::<Vec<Comment>>(&res) {
            Err(_e) => log::error!("Error parsing Vec<Comment>: {:?}", _e),
            Ok(comments_obj) => {
                for comment in comments_obj {
                    let comment_body = match comment.body {
                        Some(body) => squeeze_fit_comment_texts(&body, "```", 500, 0.6),
                        None => "".to_string(),
                    };
                    let commenter = comment.user.login;
                    let commenter_input = format!("{commenter} commented: {comment_body}");
                    all_text_from_issue.push_str(&commenter_input);

                    if all_text_from_issue.len() > 45_000 {
                        break;
                    }
                }
            }
        },
        None => {}
    };

    let sys_prompt_1 = &format!("Given the information that user '{issue_creator_name}' opened an issue titled '{issue_title}', labelled as '{labels}', your task is to analyze the content of the issue posts. Extract key details including the main problem or question raised, the environment in which the issue occurred, any steps taken by the user to address the problem, relevant discussions, and any identified solutions or pending tasks.");
    let usr_prompt_1 = &format!("Based on the GitHub issue posts: {all_text_from_issue}, please list the following key details: The main problem or question raised in the issue. The environment or conditions in which the issue occurred (e.g., hardware, OS). Any steps or actions taken by the user '{user}' or others to address the issue. Key discussions or points of view shared by participants in the issue thread. Any solutions identified, or pending tasks if the issue hasn't been resolved. The role and contribution of the user '{user}' in the issue.");
    let usr_prompt_2 = &format!("Provide a brief summary highlighting the core problem and emphasize the overarching contribution made by '{user}' to the resolution of this issue, ensuring your response stays under 128 tokens.");

    match chain_of_chat(
        sys_prompt_1,
        usr_prompt_1,
        &format!("issue_{issue_number}"),
        256,
        usr_prompt_2,
        128,
        &format!("Error generatng issue summary #{issue_number}"),
    )
    .await
    {
        Some(issue_summary) => {
            let mut out = html_url.to_string();
            out.push(' ');
            out.push_str(&issue_summary);
            return Some(out);
        }
        None => {}
    }

    None
}

pub fn squeeze_fit_commits_issues(commits: &str, issues: &str, split: f32) -> (String, String) {
    let mut commits_vec = commits.split_whitespace().collect::<Vec<&str>>();
    let commits_len = commits_vec.len();
    let mut issues_vec = issues.split_whitespace().collect::<Vec<&str>>();
    let issues_len = issues_vec.len();

    if commits_len + issues_len > 44_000 {
        let commits_to_take = (44_000 as f32 * split) as usize;
        match commits_len > commits_to_take {
            true => commits_vec.truncate(commits_to_take),
            false => {
                let issues_to_take = 44_000 - commits_len;
                issues_vec.truncate(issues_to_take);
            }
        }
    }
    (commits_vec.join(" "), issues_vec.join(" "))
}

pub fn squeeze_fit_comment_texts(
    inp_str: &str,
    quote_mark: &str,
    max_len: u16,
    split: f32,
) -> String {
    let mut body = String::new();
    let mut inside_quote = false;
    let max_len = max_len as usize;

    for line in inp_str.lines() {
        if line.contains(quote_mark) {
            inside_quote = !inside_quote;
            continue;
        }

        if !inside_quote {
            body.push_str(line);
            body.push('\n');
        }
    }

    let body_len = body.split_whitespace().count();
    let n_take_from_beginning = (max_len as f32 * split) as usize;
    let n_keep_till_end = max_len - n_take_from_beginning;
    match body_len > max_len {
        false => body,
        true => {
            let mut body_text_vec = body.split_whitespace().collect::<Vec<&str>>();
            let drain_to = std::cmp::min(body_len, max_len);
            body_text_vec.drain(n_take_from_beginning..drain_to - n_keep_till_end);
            body_text_vec.join(" ")
        }
    }
}

pub async fn analyze_commits(owner: &str, repo: &str, user_name: &str) -> Option<String> {
    let user_commits_repo_str =
        format!("https://api.github.com/repos/{owner}/{repo}/commits?author={user_name}");
    let mut commits_summaries = String::new();

    match github_http_fetch_tokenless(&user_commits_repo_str).await {
        None => log::error!("Error fetching Page of commits"),
        Some(res) => match serde_json::from_slice::<Vec<GithubCommit>>(&res) {
            Err(_e) => log::error!("Error parsing commits object: {:?}", _e),
            Ok(commits_obj) => {
                for sha in commits_obj.into_iter().map(|commit| commit.sha) {
                    let commit_patch_str =
                        format!("https://github.com/{owner}/{repo}/commit/{sha}.patch");
                    match github_http_fetch_tokenless(&commit_patch_str).await {
                        Some(res) => {
                            let text = String::from_utf8_lossy(&res).to_string();

                            let sys_prompt_1 = &format!("You are provided with a commit patch by the user {user_name} on the {repo} project. Your task is to parse this data, focusing on the following sections: the Date Line, Subject Line, Diff Files, Diff Changes, Sign-off Line, and the File Changes Summary. Extract key elements such as the date of the commit (in 'yyyy/mm/dd' format), a summary of changes, and the types of files affected, prioritizing code files, scripts, then documentation. Be particularly careful to distinguish between changes made to core code files and modifications made to documentation files, even if they contain technical content. Compile a list of the extracted key elements.");

                            let usr_prompt_1 = &format!("Based on the provided commit patch: {text}, extract and present the following key elements: the date of the commit (formatted as 'yyyy/mm/dd'), a high-level summary of the changes made, and the types of files affected. Prioritize data on changes to code files first, then scripts, and lastly documentation. Pay attention to the file types and ensure the distinction between documentation changes and core code changes, even when the documentation contains highly technical language. Please compile your findings into a list, with each key element represented as a separate item.");

                            let usr_prompt_2 = &format!("Using the key elements you extracted from the commit patch, provide a summary of the user's contributions to the project. Include the date of the commit, the types of files affected, and the overall changes made. When describing the affected files, make sure to differentiate between changes to core code files, scripts, and documentation files. Present your summary in this format: 'On (date in 'yyyy/mm/dd' format), (summary of changes). (overall impact of changes).' Please ensure your answer stayed below 128 tokens.");

                            let sha_serial = sha.chars().take(5).collect::<String>();
                            match chain_of_chat(
                                sys_prompt_1,
                                usr_prompt_1,
                                &format!("commit-{sha_serial}"),
                                256,
                                usr_prompt_2,
                                128,
                                &format!("analyze_commits-{sha_serial}"),
                            )
                            .await
                            {
                                Some(res) => {
                                    commits_summaries.push_str(&res);
                                    commits_summaries.push('\n');
                                    if commits_summaries.len() > 45_000 {
                                        break;
                                    }
                                }
                                None => continue,
                            }
                        }
                        None => continue,
                    };
                }
            }
        },
    }

    Some(commits_summaries)
}

#[derive(Debug, Deserialize, Serialize)]
struct User {
    login: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct GithubCommit {
    sha: String,
    html_url: String,
    author: User,
    committer: User,
}

pub async fn correlate_commits_issues(
    _commits_summary: &str,
    _issues_summary: &str,
) -> Option<String> {
    let (commits_summary, issues_summary) =
        squeeze_fit_commits_issues(_commits_summary, _issues_summary, 0.6);

    let sys_prompt_1 = &format!("Your task is to examine and correlate both commit logs and issue records for a specific user within a GitHub repository. Despite potential limitations in the data, such as insufficient information or difficulties in finding correlations, focus on identifying the user's top 1-3 significant contributions to the project. Consider all aspects of their contributions, from the codebase to project documentation, and describe their evolution over time. Assess the overall impact of these contributions to the project's development. Create a unique, detailed summary that highlights the scope and significance of the user's contributions, avoiding verbatim repetition from the source data. If correlations between commit logs and issue records are limited, prioritize identifying the user's top contributions. Present your summary in a clear, bullet-point format.");

    let usr_prompt_1 = &format!("Given the commit logs: {commits_summary} and issue records: {issues_summary}, analyze and identify the top 1-3 significant contributions made by the user to the project. Your task is to recognize the key areas of impact, be it in the codebase, project documentation, or other aspects, even in the presence of insufficient data or lack of direct correlations. Create a list of these significant contributions without directly replicating phrases from the source data. This list will be used in the next step to construct a detailed narrative of the user's journey in the project.");

    let usr_prompt_2 = &format!("Using the list of significant contributions identified in the previous step, create a detailed narrative that depicts the user's journey and evolution in the project. Describe the progression of these contributions over time, from their inception to their current status. Highlight the overall impact and significance of these contributions within the project's development. Your narrative should be unique and insightful, capturing the user's influence on the project. Present your findings in a clear, concise, and bullet-point format.");

    chain_of_chat(
        sys_prompt_1,
        usr_prompt_1,
        "correlate-99",
        512,
        usr_prompt_2,
        256,
        "correlate_commits_issues",
    )
    .await
}

pub async fn chain_of_chat(
    sys_prompt_1: &str,
    usr_prompt_1: &str,
    chat_id: &str,
    gen_len_1: u16,
    usr_prompt_2: &str,
    gen_len_2: u16,
    error_tag: &str,
) -> Option<String> {
    let openai = OpenAIFlows::new();

    let co_1 = ChatOptions {
        model: ChatModel::GPT35Turbo16K,
        restart: true,
        system_prompt: Some(sys_prompt_1),
        max_tokens: Some(gen_len_1),
        temperature: Some(0.7),
        ..Default::default()
    };

    match openai.chat_completion(chat_id, usr_prompt_1, &co_1).await {
        Ok(res_1) => {
            let sys_prompt_2 = serde_json::json!([{"role": "system", "content": sys_prompt_1},
    {"role": "user", "content": usr_prompt_1},
    {"role": "assistant", "content": &res_1.choice}])
            .to_string();

            let co_2 = ChatOptions {
                model: ChatModel::GPT35Turbo16K,
                restart: false,
                system_prompt: Some(&sys_prompt_2),
                max_tokens: Some(gen_len_2),
                temperature: Some(0.7),
                ..Default::default()
            };
            match openai.chat_completion(chat_id, usr_prompt_2, &co_2).await {
                Ok(res_2) => {
                    if res_2.choice.len() < 10 {
                        log::error!(
                            "{}, GPT generation went sideway: {:?}",
                            error_tag,
                            res_2.choice
                        );
                        return None;
                    }
                    return Some(res_2.choice);
                }
                Err(_e) => log::error!("{}, Step 2 GPT generation error {:?}", error_tag, _e),
            };
        }
        Err(_e) => log::error!("{}, Step 1 GPT generation error {:?}", error_tag, _e),
    }

    None
}

pub async fn github_http_fetch_tokenless(url: &str) -> Option<Vec<u8>> {
    let url = Uri::try_from(url).unwrap();
    let mut writer = Vec::new();

    match Request::new(&url)
        .method(Method::GET)
        .header("User-Agent", "flows-network connector")
        .header("Content-Type", "application/vnd.github.v3+json")
        .send(&mut writer)
    {
        Ok(res) => {
            if !res.status_code().is_success() {
                log::error!("Github http error {:?}", res.status_code());
                return None;
            };

            return Some(writer);
        }
        Err(_e) => {
            log::error!("Error getting response from Github: {:?}", _e);
        }
    }

    None
}
