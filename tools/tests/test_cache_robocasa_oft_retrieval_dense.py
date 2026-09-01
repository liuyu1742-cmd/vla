from tools.cache_robocasa_oft_retrieval_dense import select_shortest_per_task_class


def test_select_shortest_per_task_class_returns_one_episode_each():
    episodes = [
        {"task_class": "A", "frames": 20, "episode_index": 2},
        {"task_class": "A", "frames": 10, "episode_index": 1},
        {"task_class": "B", "frames": 15, "episode_index": 3},
    ]

    selected = select_shortest_per_task_class(episodes)

    assert [(item["task_class"], item["episode_index"]) for item in selected] == [
        ("A", 1),
        ("B", 3),
    ]
