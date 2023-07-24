use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDate, Utc};
use dotenv::dotenv;
use flowsnet_platform_sdk::logger;
use github_flows::{
    get_octo, octocrab,
    octocrab::{
        models::issues::{Comment, Issue},
        Result as OctoResult,
    },
    GithubLogin,
};
use http_req::{request::Method, request::Request, uri::Uri};
use log;
use openai_flows::{
    chat::{self, ChatMessage, ChatModel, ChatOptions},
    OpenAIFlows,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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

    let mut out = String::from("placeholder");
    if sm.text.contains(&trigger_word) {
        // let mut issues_summaries = String::new();
        let mut output = String::new();
        if let Ok(issues) = get_issues(owner, repo, user_name).await {
            for issue in issues {
                if let Some(body) = analyze_issue(owner, repo, user_name, issue).await {
                    // send_message_to_channel("ik8", "ch_in", body.to_string()).await;
                    break;
                    // issues_summaries.push_str(&body);
                    // issues_summaries.push_str("\n");
                }
            }
            // send_message_to_channel("ik8", "ch_out", format!("issues_count: {count}   {output}"))
            //     .await;
        }
    }
}
#[derive(Debug, Deserialize)]
struct Page<T> {
    pub items: Vec<T>,
    pub incomplete_results: Option<bool>,
    pub total_count: Option<u64>,
    // pub next: Option<String>,
    // pub prev: Option<String>,
    // pub first: Option<String>,
    // pub last: Option<String>,
}
pub async fn get_issues(owner: &str, repo: &str, user: &str) -> anyhow::Result<Vec<Issue>> {
    let github_token = env::var("github_token").unwrap_or("fake-token".to_string());
    let query = format!("repo:{}/{} involves:{}", owner, repo, user);
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

        let url = Uri::try_from(url_str.as_str()).unwrap();
        let mut writer = Vec::new();

        match Request::new(&url)
            .method(Method::GET)
            .header("User-Agent", "flows-network connector")
            .header("Content-Type", "application/vnd.github.v3+json")
            .header("Authorization", &format!("Bearer {github_token}")) // add the token to your request
            .send(&mut writer)
        {
            Ok(res) => {
                if !res.status_code().is_success() {
                    continue;
                };

                let response: Result<Page<Issue>, _> = serde_json::from_slice(&writer);

                match response {
                    Err(_e) => {
                        continue;
                    }

                    Ok(search_result) => {
                        if total_pages.is_none() {
                            if let Some(count) = search_result.total_count {
                                total_pages = Some((count / 30) as usize + 1);
                            }
                        }

                        for issue in search_result.items {
                            out.push(issue.clone());
                        }
                    }
                }
            }
            Err(_e) => {
                continue;
            }
        }
    }

    Ok(out)
}

pub async fn analyze_issue(owner: &str, repo: &str, user: &str, issue: Issue) -> Option<String> {
    let openai = OpenAIFlows::new();
    let github_token = env::var("github_token").unwrap_or("fake-token".to_string());

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

    let url = Uri::try_from(url_str.as_str()).unwrap();
    let mut writer = Vec::new();

    match Request::new(&url)
        .method(Method::GET)
        .header("User-Agent", "flows-network connector")
        .header("Content-Type", "application/vnd.github.v3+json")
        .header("Authorization", &format!("Bearer {}", github_token))
        .send(&mut writer)
    {
        Ok(res) => {
            if !res.status_code().is_success() {
                log::error!("Github http error {:?}", res.status_code());
            };

            let response: Result<Vec<Comment>, _> = serde_json::from_slice(&writer);

            match response {
                Err(_e) => log::error!("Github response parse error {:?}", _e),

                Ok(comments) => {
                    for comment in comments {
                        let comment_body = match comment.body {
                            Some(body) => squeeze_fit_comment_texts(&body, "```", 500, 0.6),
                            None => "".to_string(),
                        };
                        let commenter = comment.user.login;

                        let commenter_input = format!("{commenter} commented: {comment_body}");
                        all_text_from_issue.push_str(&commenter_input);

                        if all_text_from_issue.len() > 55_000 {
                            break;
                        }
                    }
                }
            }
        }
        Err(_e) => log::error!("Error getting GitHub response {:?}", _e),
    }

    let mut out = issue_date;
    let sys_prompt_1 = &format!("Given the information that user '{issue_creator_name}' opened an issue titled '{issue_title}', labelled as '{labels}', your task is to analyze the content of the issue posts. Extract key details including the main problem or question raised, the environment in which the issue occurred, any steps taken by the user to address the problem, relevant discussions, and any identified solutions or pending tasks.");
    let usr_prompt_1 = &format!("Based on the GitHub issue posts: {all_text_from_issue}, please list the following key details: The main problem or question raised in the issue. The environment or conditions in which the issue occurred (e.g., hardware, OS). Any steps or actions taken by the user '{user}' or others to address the issue. Key discussions or points of view shared by participants in the issue thread. Any solutions identified, or pending tasks if the issue hasn't been resolved. The role and contribution of the user '{user}' in the issue.");
    let chat_id = format!("issue_{issue_number}");

    let co_1 = ChatOptions {
        model: ChatModel::GPT35Turbo16K,
        restart: true,
        system_prompt: Some(sys_prompt_1),
        max_tokens: Some(256),
        temperature: Some(0.7),
        ..Default::default()
    };

    match openai.chat_completion(&chat_id, usr_prompt_1, &co_1).await {
        Ok(res_1) => {
            send_message_to_channel("ik8", "ch_mid", res_1.choice.clone()).await;

            let system_obj_1 = serde_json::json!(
                {"role": "system", "content": sys_prompt_1}
            );

            let user_obj_1 = serde_json::json!(
                {"role": "user", "content": usr_prompt_1}
            );
            let assistant_obj = serde_json::json!(
                {"role": "assistant", "content": &res_1.choice}
            );
            let sys_prompt_2 =
                serde_json::json!([system_obj_1, user_obj_1, assistant_obj]).to_string();
            let usr_prompt_2 = &format!("Based on the key details listed in the previous step, provide a high-level summary of the issue <Brief summary of the main problem, and current status of the issue>. Highlight the role and contribution of '{user}'");

            let co_2 = ChatOptions {
                model: ChatModel::GPT35Turbo16K,
                restart: false,
                system_prompt: Some(&sys_prompt_2),
                max_tokens: Some(128),
                temperature: Some(0.7),
                ..Default::default()
            };
            match openai.chat_completion(&chat_id, usr_prompt_2, &co_2).await {
                Ok(res_2) => {
                    send_message_to_channel("ik8", "ch_out", res_2.choice.clone()).await;

                    if res_2.choice.len() < 10 {
                        return None;
                    }
                    out.push(' ');
                    out.push_str(&html_url);
                    out.push(' ');
                    out.push_str(&res_2.choice);
                    println!("{:?}", out);
                }
                Err(_e) => log::error!("Step 2 GPT error {:?}", _e),

            };
        }
        Err(_e) => log::error!("Step 1 GPT error {:?}", _e),

    }

    Some(out)
}

pub fn squeeze_fit_commits_issues(commits: &str, issues: &str, split: f32) -> (String, String) {
    let mut commits_vec = commits.split_whitespace().collect::<Vec<&str>>();
    let commits_len = commits_vec.len();
    let mut issues_vec = issues.split_whitespace().collect::<Vec<&str>>();
    let issues_len = issues_vec.len();

    if commits_len + issues_len > 5500 {
        let commits_to_take = (5500 as f32 * split) as usize;
        match commits_len > commits_to_take {
            true => commits_vec.truncate(commits_to_take),
            false => {
                let issues_to_take = 5500 - commits_len;
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
