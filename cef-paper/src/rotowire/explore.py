# %%
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk

# %%
def process_rotowire_entry(entry):
    # --- Helper to format team stats ---
    def get_team_stats(line_data, team_name):
        # These keys might vary slightly, but usually follow this pattern
        # We extract high-level summary stats
        stats = {
            "Team": team_name,
            "Points": line_data.get("TEAM-PTS", "N/A"),
            "Assists": line_data.get("TEAM-AST", "N/A"),
            "Rebounds": line_data.get("TEAM-REB", "N/A"), # frequent key in this dataset
            "FG%": line_data.get("TEAM-FG_PCT", "N/A"),
            "3P%": line_data.get("TEAM-FG3_PCT", "N/A"),
            "Wins": line_data.get("TEAM-WINS", "N/A"), # Season wins if available
            "Losses": line_data.get("TEAM-LOSSES", "N/A")
        }
        return stats

    # 1. Process Team Summaries (The "Line" columns)
    home_stats = get_team_stats(entry['home_line'], entry['home_name'])
    vis_stats = get_team_stats(entry['vis_line'], entry['vis_name'])
    
    # Construct the Header with the Scoreboard
    output_text = f"GAME SUMMARY: {entry['vis_city']} {entry['vis_name']} @ {entry['home_city']} {entry['home_name']}\n"
    output_text += f"Date: {entry['day']}\n"
    output_text += f"Home Team: {entry['home_city']} {entry['home_name']}\n"
    output_text += f"Away Team: {entry['vis_city']} {entry['vis_name']}\n\n"
    
    output_text += "### FINAL SCOREBOARD\n"
    output_text += f"| Team | Final Score | FG% | 3P% | Assists | Wins-Losses |\n"
    output_text += f"| --- | --- | --- | --- | --- | --- |\n"
    
    # Helper to create row string
    def make_row(s):
        return f"| {s['Team']} | {s['Points']} | {s['FG%']} | {s['3P%']} | {s['Assists']} | {s['Wins']}-{s['Losses']} |"

    output_text += make_row(vis_stats) + "\n"
    output_text += make_row(home_stats) + "\n\n"

    # 2. Process Player Stats (The "Box Score" - same logic as before)
    box_score = entry['box_score']
    valid_indices = [k for k, v in box_score['PLAYER_NAME'].items() if v is not None and v != 'N/A']
    
    players = []
    for idx in valid_indices:
        player = {}
        for stat_name, stat_values in box_score.items():
            player[stat_name] = stat_values.get(idx, "N/A")
        players.append(player)
        
    teams = {}
    for p in players:
        team_city = p.get('TEAM_CITY', 'Unknown')
        if team_city not in teams:
            teams[team_city] = []
        teams[team_city].append(p)
        
    for city, roster in teams.items():
        output_text += f"### Player Stats for {city}\n"
        cols = ["PLAYER_NAME", "START_POSITION", "MIN", "PTS", "AST", "REB", "STL", "BLK", "TO", "FGM", "FGA", "FG3M"]
        output_text += "| " + " | ".join(cols) + " |\n"
        output_text += "| " + " | ".join(["---"] * len(cols)) + " |\n"
        for p in roster:
            row = [str(p.get(c, "0")) for c in cols]
            output_text += "| " + " | ".join(row) + " |\n"
        output_text += "\n"        
    return {"src": output_text, "trg": " ".join(entry["summary"])}
# %%
ds = load_dataset("json", data_files="/home/yathagata/cef-translation/data/boxscore-data/rotowire/test.json")
ds["test"] = ds["train"]
del ds["train"]

ds2 = load_dataset("json", data_files="/home/yathagata/cef-translation/data/boxscore-data/sbnation/test.json")
ds2["test"] = ds2["train"]
del ds2["train"]

# %%
# %%
ds["test"] = concatenate_datasets([ds["test"], ds2["test"]])
ds = ds.map(process_rotowire_entry, num_proc=16, remove_columns=['home_name', 'box_score', 'home_city', 'vis_name', 'summary', 'vis_line', 'vis_city', 'day', 'home_line'])

# %%
ds = ds.map(lambda x, idx: {"id": idx}, with_indices=True)
# %%
ds.save_to_disk("/data/cef/boxscore/boxscore_original")
# %%
ds = load_from_disk("/data/cef/boxscore/boxscore_original")
# %%
