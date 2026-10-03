from sleep_monitor.config.settings import Settings


def test_default_config():
    """Test that default configuration can be loaded."""
    settings = Settings()
    assert settings.video.frame_sample_rate == 2
    assert settings.perception.detection_model == "yolov8n.pt"
    assert settings.target_person.mode == "auto"


def test_yaml_config(tmp_path):
    """Test that configuration can be loaded from YAML."""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text("""
video:
  frame_sample_rate: 5
target_person:
  mode: manual
""")
    settings = Settings.from_yaml(yaml_file)
    assert settings.video.frame_sample_rate == 5
    assert settings.target_person.mode == "manual"
    # Fallback to default
    assert settings.perception.detection_model == "yolov8n.pt"
