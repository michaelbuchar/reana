# -*- coding: utf-8 -*-
#
# This file is part of REANA
# Copyright (C) 2024, 2026 CERN.
#
# REANA is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.

"""REANA CLI cluster command tests."""

from __future__ import absolute_import, print_function

import copy
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from mock import patch
from unittest.mock import call


def _helm_install_command(values_dict):
    """Render the expected inline Helm values document."""
    values_yaml = yaml.dump(values_dict, width=100000) if values_dict else ""
    return (
        "cat <<EOF | helm install reana helm/reana -n default --create-namespace --wait -f -\n"
        f"{values_yaml}\n"
        "EOF"
    )


@pytest.mark.parametrize(
    "mode, shared_storage_backend, values_files, expected",
    [
        (
            "latest",
            "hostpath",
            (),
            ("helm/configurations/values-dev.yaml",),
        ),
        ("releasehelm", "hostpath", (), ()),
        (
            "latest",
            "cephfs",
            (),
            (
                "helm/configurations/values-dev.yaml",
                "helm/configurations/values-dev-cephfs.yaml",
            ),
        ),
        (
            "latest",
            "cephfs",
            ("custom-values.yaml",),
            (
                "custom-values.yaml",
                "helm/configurations/values-dev-cephfs.yaml",
            ),
        ),
        (
            "releasehelm",
            "cephfs",
            (),
            ("helm/configurations/values-dev-cephfs.yaml",),
        ),
        (
            "releasehelm",
            "cephfs",
            ("custom-values.yaml",),
            (
                "custom-values.yaml",
                "helm/configurations/values-dev-cephfs.yaml",
            ),
        ),
        (
            "latest",
            "cephfs",
            (
                "helm/configurations/values-dev-cephfs.yaml",
                "custom-values.yaml",
            ),
            (
                "custom-values.yaml",
                "helm/configurations/values-dev-cephfs.yaml",
            ),
        ),
    ],
)
def test_default_cluster_values_files(
    mode, shared_storage_backend, values_files, expected
):
    """Cluster deploy should layer local dev values files predictably."""
    from reana.reana_dev.cluster import default_cluster_values_files

    assert (
        default_cluster_values_files(mode, shared_storage_backend, values_files)
        == expected
    )


def test_merge_values_dicts_recurses_and_replaces_whole_values():
    """Values overlays should recurse into mappings and replace other values."""
    from reana.reana_dev.cluster import merge_values_dicts

    base = {
        "nested": {"keep": 1, "replace": {"old": True}},
        "replace": ["old"],
    }
    overlay = {
        "nested": {"replace": "new", "add": 2},
        "replace": {"new": True},
    }

    assert merge_values_dicts(base, overlay) == {
        "nested": {"keep": 1, "replace": "new", "add": 2},
        "replace": {"new": True},
    }


@patch("reana.reana_dev.cluster.get_srcdir")
def test_load_cluster_values_merges_real_dev_and_cephfs_files(get_srcdir_mock):
    """The shipped CephFS overlay should merge with real development secrets."""
    from reana.reana_dev.cluster import load_cluster_values

    get_srcdir_mock.return_value = str(Path(__file__).parent.parent)
    values = load_cluster_values(
        (
            "helm/configurations/values-dev.yaml",
            "helm/configurations/values-dev-cephfs.yaml",
        )
    )

    assert values["secrets"]["database"]["user"]
    assert values["shared_storage"] == {
        "backend": "cephfs",
        "fs_group": 0,
        "storage_class_name": "rook-cephfs",
    }
    assert values["infrastructure_storage"]["hostpath"]["root_path"] == (
        "/var/reana-infrastructure"
    )
    assert (
        values["node_label_infrastructuredb"]
        == "reana.io/infrastructure-storage=cephfs"
    )
    assert (
        values["node_label_infrastructuremq"]
        == "reana.io/infrastructure-storage=cephfs"
    )


@patch("reana.reana_dev.cluster.display_message")
def test_validate_shared_storage_backend_rejects_non_kind_cephfs(
    display_message_mock,
):
    """Local CephFS must remain explicitly scoped to Kind for now."""
    from reana.reana_dev.cluster import validate_shared_storage_backend

    with pytest.raises(SystemExit):
        validate_shared_storage_backend("colima/k3s", "cephfs")

    display_message_mock.assert_called_once_with(
        "[ERROR] Local CephFS shared storage is currently supported only with --kubernetes kind. Exiting.",
        "reana",
    )


def test_validate_multi_node_mounts_accepts_cephfs_without_host_mounts():
    """CephFS itself satisfies the multi-node shared-workspace requirement."""
    from reana.reana_dev.cluster import validate_multi_node_mounts

    validate_multi_node_mounts([], 2, "cephfs")
    with pytest.raises(SystemExit):
        validate_multi_node_mounts([], 2, "hostpath")


def test_cephfs_lifecycle_state_round_trip(tmp_path, monkeypatch):
    """Deletion should discover backend and owned devices from host state."""
    from reana.reana_dev.cluster import (
        cephfs_state_file,
        load_cephfs_state,
        selected_shared_storage_backend,
    )

    monkeypatch.setenv("REANA_DEV_STATE_DIR", str(tmp_path))
    Path(cephfs_state_file()).write_text(
        "backend\tcephfs\n"
        "node_name\tkind-control-plane\n"
        "node_container\tkind-control-plane\n"
        "device\t/dev/loop3\t/var/lib/rook-dev/osd-0.img\n"
    )

    state = load_cephfs_state()
    assert state["backend"] == "cephfs"
    assert state["devices"] == [
        {
            "path": "/dev/loop3",
            "backing_file": "/var/lib/rook-dev/osd-0.img",
        }
    ]
    assert selected_shared_storage_backend(None) == ("cephfs", state)


@pytest.mark.parametrize(
    "options, initial_values, expected_values_files, expected_final_values, run_command_side_effects, exit_code",
    [
        (
            [
                "--admin-email",
                "john.doe@reana.io",
                "--admin-password",
                "admin",
                "--values",
                "alternative-values-dev.yaml",
                "--mode",
                "debug",
                "--exclude-components",
                "reana-ui,reana-workflow-controller",
            ],
            {},
            ("alternative-values-dev.yaml",),
            {
                "components": {"reana_ui": {"enabled": False}},
                "debug": {"enabled": True},
            },
            [None] * 6,
            0,
        ),
        (
            [
                "--admin-email",
                "john.doe@reana.io",
                "--admin-password",
                "admin",
            ],
            {
                "debug": {"enabled": True},
                "components": {
                    "reana_workflow_controller": {
                        "environment": {"REANA_OPENSEARCH_ENABLED": True}
                    }
                },
            },
            ("helm/configurations/values-dev.yaml",),
            {
                "debug": {"enabled": True},
                "components": {
                    "reana_workflow_controller": {
                        "environment": {"REANA_OPENSEARCH_ENABLED": True}
                    }
                },
            },
            [None] * 4,
            0,
        ),
        (
            [
                "--admin-email",
                "john.doe@reana.io",
                "--admin-password",
                "admin",
                "--shared-storage-backend",
                "cephfs",
                "--values",
                "custom-values.yaml",
            ],
            {"custom": {"enabled": True}},
            (
                "custom-values.yaml",
                "helm/configurations/values-dev-cephfs.yaml",
            ),
            {"custom": {"enabled": True}},
            [None] * 5,
            0,
        ),
        (
            [
                "--admin-email",
                "john.doe@reana.io",
                "--admin-password",
                "admin",
                "--shared-storage-backend",
                "cephfs",
                "--mode",
                "releasehelm",
            ],
            {
                "shared_storage": {
                    "backend": "cephfs",
                    "storage_class_name": "rook-cephfs",
                }
            },
            ("helm/configurations/values-dev-cephfs.yaml",),
            {
                "shared_storage": {
                    "backend": "cephfs",
                    "storage_class_name": "rook-cephfs",
                }
            },
            [None] * 5,
            0,
        ),
        (
            [
                "--admin-email",
                "john.doe@reana.io",
                "--admin-password",
                "admin",
                "--mode",
                "releasehelm",
            ],
            {},
            None,
            {},
            [None] * 4,
            0,
        ),
        (
            [
                "--admin-email",
                "john.doe@reana.io",
                "--admin-password",
                "admin",
                "--shared-storage-backend",
                "cephfs",
            ],
            {
                "shared_storage": {
                    "backend": "cephfs",
                    "fs_group": 0,
                    "storage_class_name": "rook-cephfs",
                }
            },
            (
                "helm/configurations/values-dev.yaml",
                "helm/configurations/values-dev-cephfs.yaml",
            ),
            {
                "shared_storage": {
                    "backend": "cephfs",
                    "fs_group": 0,
                    "storage_class_name": "rook-cephfs",
                }
            },
            [None] * 5,
            0,
        ),
        (
            [
                "--admin-email",
                "john.doe@reana.io",
                "--admin-password",
                "admin",
            ],
            {},
            ("helm/configurations/values-dev.yaml",),
            {},
            [None, None, None, ValueError()],
            1,
        ),
    ],
)
@patch(
    "reana.reana_dev.cluster.cephfs_state_file",
    new=lambda: "/state/kind-cephfs.state",
)
@patch("reana.reana_dev.cluster.get_srcdir")
@patch("reana.reana_dev.cluster.load_cluster_values")
@patch("reana.reana_dev.cluster.run_command")
def test_cluster_deploy(
    run_command_mock,
    load_cluster_values_mock,
    get_srcdir_mock,
    options,
    initial_values,
    expected_values_files,
    expected_final_values,
    run_command_side_effects,
    exit_code,
):
    """Test cluster-deploy command."""
    from reana.reana_dev.cluster import cluster_deploy

    run_command_mock.side_effect = run_command_side_effects
    load_cluster_values_mock.return_value = copy.deepcopy(initial_values)
    get_srcdir_mock.return_value = "/code/src/reana"

    runner = CliRunner()
    result = runner.invoke(cluster_deploy, options)

    if expected_values_files is None:
        load_cluster_values_mock.assert_not_called()
    else:
        load_cluster_values_mock.assert_called_once_with(expected_values_files)

    expected_run_command_calls = []
    if (
        "--shared-storage-backend" in options
        and options[options.index("--shared-storage-backend") + 1] == "cephfs"
    ):
        expected_run_command_calls.append(
            call(
                [
                    "/bin/sh",
                    "/code/src/reana/scripts/deploy-kind-rook-cephfs.sh",
                    "--check-only",
                    "/state/kind-cephfs.state",
                ],
                "reana",
            )
        )
    if "--mode" in options and options[options.index("--mode") + 1] == "debug":
        expected_run_command_calls.extend(
            [
                call("reana-dev python-install-eggs", "reana"),
                call("reana-dev git-submodule --update", "reana"),
            ]
        )
    expected_run_command_calls.extend(
        [
            call("helm dep update helm/reana", "reana"),
            call(_helm_install_command(expected_final_values), "reana"),
            call("kubectl config set-context --current --namespace=default", "reana"),
            call(
                "/code/src/reana/scripts/create-admin-user.sh default reana john.doe@reana.io admin",
                "reana",
            ),
        ]
    )

    assert run_command_mock.call_args_list == expected_run_command_calls
    assert result.exit_code == exit_code


def test_cluster_deploy_help_describes_effective_values_defaults():
    """The CLI should explain implicit values and storage-overlay precedence."""
    from reana.reana_dev.cluster import cluster_deploy

    result = CliRunner().invoke(cluster_deploy, ["--help"])

    assert result.exit_code == 0
    normalized_help = " ".join(result.output.split())
    assert "values-dev.yaml is used except in releasehelm mode" in normalized_help
    assert "shared-storage overlay is applied last" in normalized_help


@patch(
    "reana.reana_dev.cluster.cephfs_state_file",
    new=lambda: "/state/kind-cephfs.state",
)
@patch("reana.reana_dev.cluster.get_srcdir")
@patch("reana.reana_dev.cluster.run_command")
def test_cluster_create_cephfs_invokes_kind_helpers(run_command_mock, get_srcdir_mock):
    """Kind cluster creation should own the local CephFS bootstrap steps."""
    from reana.reana_dev.cluster import cluster_create

    run_command_mock.side_effect = [
        "",
        None,
        None,
        "kind-control-plane",
        None,
        None,
        None,
        None,
    ]
    get_srcdir_mock.return_value = "/code/src/reana"

    result = CliRunner().invoke(
        cluster_create,
        ["--shared-storage-backend", "cephfs"],
    )

    assert result.exit_code == 0
    assert run_command_mock.call_args_list[0] == call(
        "docker version", return_output=True
    )
    cluster_create_cmd = run_command_mock.call_args_list[1]
    assert cluster_create_cmd.args[1] == "reana"
    assert "kind create cluster" in cluster_create_cmd.args[0]
    assert "containerPort: 30080" in cluster_create_cmd.args[0]
    assert "containerPort: 30443" in cluster_create_cmd.args[0]
    assert (
        'node-labels: "ingress-ready=true,reana.io/infrastructure-storage=cephfs"'
        in cluster_create_cmd.args[0]
    )
    assert run_command_mock.call_args_list[2:] == [
        call(
            "docker exec kind-control-plane sh -c 'mkdir -p /var/reana && chmod g+rwx /var/reana'",
            "reana",
        ),
        call(
            [
                "kubectl",
                "get",
                "nodes",
                "-l",
                "reana.io/infrastructure-storage=cephfs",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            "reana",
            return_output=True,
        ),
        call(
            [
                "/bin/sh",
                "/code/src/reana/scripts/setup-kind-rook-loop-devices.sh",
                "kind-control-plane",
                "kind-control-plane",
                "/state/kind-cephfs.state",
            ],
            "reana",
        ),
        call(
            [
                "/bin/sh",
                "/code/src/reana/scripts/deploy-kind-rook-cephfs.sh",
                "/state/kind-cephfs.state",
            ],
            "reana",
        ),
        call("reana-dev docker-pull -c reana", "reana"),
        call("reana-dev kind-load-docker-image -c reana", "reana"),
    ]


@patch(
    "reana.reana_dev.cluster.selected_shared_storage_backend",
    return_value=("cephfs", {"backend": "cephfs"}),
)
@patch(
    "reana.reana_dev.cluster.cephfs_state_file",
    new=lambda: "/state/kind-cephfs.state",
)
@patch("reana.reana_dev.cluster.get_srcdir")
@patch("reana.reana_dev.cluster.run_command")
def test_cluster_delete_cephfs_invokes_kind_helpers(
    run_command_mock, get_srcdir_mock, selected_backend_mock
):
    """Kind cluster deletion should tear local CephFS down before removing Kind."""
    from reana.reana_dev.cluster import cluster_delete

    get_srcdir_mock.return_value = "/code/src/reana"

    result = CliRunner().invoke(
        cluster_delete,
        [],
    )

    assert result.exit_code == 0
    selected_backend_mock.assert_called_once_with(None)
    assert run_command_mock.call_args_list == [
        call(
            [
                "helm",
                "uninstall",
                "reana",
                "-n",
                "default",
                "--ignore-not-found",
                "--wait",
                "--timeout",
                "5m",
            ],
            "reana",
            exit_on_error=False,
        ),
        call(
            [
                "/bin/sh",
                "/code/src/reana/scripts/undeploy-kind-rook-cephfs.sh",
                "/state/kind-cephfs.state",
            ],
            "reana",
            exit_on_error=False,
        ),
        call(
            [
                "/bin/sh",
                "/code/src/reana/scripts/cleanup-kind-rook-loop-devices.sh",
                "/state/kind-cephfs.state",
            ],
            "reana",
            exit_on_error=False,
        ),
        call("kind delete cluster", "reana", exit_on_error=False),
    ]


@patch(
    "reana.reana_dev.cluster.selected_shared_storage_backend",
    return_value=("cephfs", {"backend": "cephfs"}),
)
@patch(
    "reana.reana_dev.cluster.cephfs_state_file",
    new=lambda: "/state/kind-cephfs.state",
)
@patch("reana.reana_dev.cluster.get_srcdir", return_value="/code/src/reana")
@patch("reana.reana_dev.cluster.run_command")
def test_cluster_delete_attempts_every_phase_after_cleanup_failure(
    run_command_mock, get_srcdir_mock, selected_backend_mock
):
    """A failed optional cleanup phase must not prevent Kind deletion."""
    from reana.reana_dev.cluster import cluster_delete

    run_command_mock.side_effect = [
        None,
        subprocess.CalledProcessError(1, "rook teardown"),
        None,
        None,
    ]

    result = CliRunner().invoke(cluster_delete, [])

    assert result.exit_code == 1
    assert run_command_mock.call_count == 4
    assert run_command_mock.call_args_list[-1] == call(
        "kind delete cluster", "reana", exit_on_error=False
    )
    assert "Rook teardown" in result.output
