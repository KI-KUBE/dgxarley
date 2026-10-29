"""Pytest collection settings for the tests directory."""

# tests/fixtures/ holds Ansible fixtures (fake collections, custom app dirs), not test modules.
collect_ignore_glob = ["fixtures/*"]
