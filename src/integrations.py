"""
Integration layer for the Apex Harness API.

Provides client interfaces to:
  - ApexGraphSwarm: graph intelligence + multi-agent orchestration
  - GRC_Claw: ISO 42001 governance, risk, and compliance
  - Apex_ULL: C++20 + Rust + Python kernels (106 components)
  - Data Center Commander: DC lifecycle management

Each integration exposes a clean async interface with health checks,
capability discovery, and operation execution.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


# ---------------------------------------------------------------------------
# Base Integration
# ---------------------------------------------------------------------------


@dataclass
class IntegrationHealth:
    """Health status of an integration."""

    name: str
    healthy: bool
    latency_ms: float = 0.0
    last_check: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationResult:
    """Result from an integration operation."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseIntegration(ABC):
    """Abstract base class for all integrations."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout_s,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> IntegrationHealth:
        """Check the health of the integration."""
        start = time.monotonic()
        try:
            client = await self._get_client()
            response = await client.get("/health")
            latency = (time.monotonic() - start) * 1000
            return IntegrationHealth(
                name=self.name,
                healthy=response.status_code == 200,
                latency_ms=latency,
                last_check=str(time.time()),
                details={"status_code": response.status_code},
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return IntegrationHealth(
                name=self.name,
                healthy=False,
                latency_ms=latency,
                last_check=str(time.time()),
                details={"error": str(exc)},
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Integration name."""
        ...

    @abstractmethod
    async def execute(self, operation: str, params: dict[str, Any]) -> IntegrationResult:
        """Execute an operation on this integration."""
        ...


# ---------------------------------------------------------------------------
# ApexGraphSwarm Integration
# ---------------------------------------------------------------------------


class ApexGraphSwarmIntegration(BaseIntegration):
    """
    Integration with ApexGraphSwarm for graph intelligence and multi-agent orchestration.

    Capabilities:
      - Graph construction and querying
      - Multi-agent task dispatch and coordination
      - Agent status and capability discovery
      - Graph-based task routing (Laya)
    """

    @property
    def name(self) -> str:
        return "apex_graph_swarm"

    async def execute(self, operation: str, params: dict[str, Any]) -> IntegrationResult:
        """Execute a graph swarm operation."""
        try:
            client = await self._get_client()

            if operation == "route":
                # Laya-based task routing
                response = await client.post("/v1/route", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "dispatch":
                # Dispatch a task to the swarm
                response = await client.post("/v1/dispatch", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "agent_status":
                # Get status of all agents in the swarm
                response = await client.get("/v1/agents")
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "graph_query":
                # Query the knowledge graph
                response = await client.post("/v1/graph/query", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "graph_build":
                # Build/update the knowledge graph
                response = await client.post("/v1/graph/build", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unknown operation: {operation}",
                )

        except httpx.HTTPStatusError as exc:
            return IntegrationResult(
                success=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.text}",
                metadata={"status_code": exc.response.status_code},
            )
        except Exception as exc:
            return IntegrationResult(success=False, error=str(exc))

    async def route_task(self, task: str, context: dict[str, Any]) -> IntegrationResult:
        """Route a task using Laya (graph-based routing)."""
        return await self.execute("route", {"task": task, "context": context})

    async def get_agents(self) -> IntegrationResult:
        """Get all agents in the swarm."""
        return await self.execute("agent_status", {})

    async def dispatch_task(self, agent_id: str, task: str, parameters: dict[str, Any]) -> IntegrationResult:
        """Dispatch a task to a specific agent."""
        return await self.execute("dispatch", {
            "agent_id": agent_id,
            "task": task,
            "parameters": parameters,
        })


# ---------------------------------------------------------------------------
# GRC_Claw Integration (ISO 42001 Governance)
# ---------------------------------------------------------------------------


class GRCClawIntegration(BaseIntegration):
    """
    Integration with GRC_Claw for ISO 42001 governance, risk, and compliance.

    Capabilities:
      - Governance policy checks
      - Risk assessment
      - Compliance auditing
      - Control mapping
    """

    @property
    def name(self) -> str:
        return "grc_claw"

    async def execute(self, operation: str, params: dict[str, Any]) -> IntegrationResult:
        """Execute a governance operation."""
        try:
            client = await self._get_client()

            if operation == "check":
                # Run a governance check
                response = await client.post("/v1/governance/check", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "risk_assessment":
                # Perform risk assessment
                response = await client.post("/v1/risk/assess", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "audit":
                # Run a compliance audit
                response = await client.post("/v1/audit", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "controls":
                # List applicable controls
                response = await client.get("/v1/controls", params=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "policies":
                # List governance policies
                response = await client.get("/v1/policies", params=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unknown operation: {operation}",
                )

        except httpx.HTTPStatusError as exc:
            return IntegrationResult(
                success=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.text}",
                metadata={"status_code": exc.response.status_code},
            )
        except Exception as exc:
            return IntegrationResult(success=False, error=str(exc))

    async def check_compliance(
        self,
        framework: str,
        target: str,
        context: dict[str, Any],
    ) -> IntegrationResult:
        """Check compliance against a governance framework."""
        return await self.execute("check", {
            "framework": framework,
            "target": target,
            "context": context,
        })

    async def assess_risk(self, target: str, context: dict[str, Any]) -> IntegrationResult:
        """Assess risk for a target system or process."""
        return await self.execute("risk_assessment", {
            "target": target,
            "context": context,
        })


# ---------------------------------------------------------------------------
# Apex_ULL Integration (C++20 + Rust + Python Kernels)
# ---------------------------------------------------------------------------


class ApexULLIntegration(BaseIntegration):
    """
    Integration with Apex_ULL for high-performance kernel operations.

    106 components spanning C++20, Rust, and Python kernels for:
      - Data processing and transformation
      - ML inference acceleration
      - Cryptographic operations
      - Graph computation kernels
      - Memory management
    """

    @property
    def name(self) -> str:
        return "apex_ull"

    async def execute(self, operation: str, params: dict[str, Any]) -> IntegrationResult:
        """Execute a kernel operation."""
        try:
            client = await self._get_client()

            if operation == "invoke":
                # Invoke a specific kernel component
                component = params.pop("component", None)
                if not component:
                    return IntegrationResult(success=False, error="component is required")
                response = await client.post(f"/v1/kernels/{component}/invoke", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "list_components":
                # List all available kernel components
                response = await client.get("/v1/kernels")
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "component_info":
                # Get info about a specific component
                component = params.pop("component", None)
                if not component:
                    return IntegrationResult(success=False, error="component is required")
                response = await client.get(f"/v1/kernels/{component}")
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "benchmark":
                # Benchmark a component
                component = params.pop("component", None)
                if not component:
                    return IntegrationResult(success=False, error="component is required")
                response = await client.post(f"/v1/kernels/{component}/benchmark", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "batch":
                # Batch invoke multiple components
                response = await client.post("/v1/kernels/batch", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unknown operation: {operation}",
                )

        except httpx.HTTPStatusError as exc:
            return IntegrationResult(
                success=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.text}",
                metadata={"status_code": exc.response.status_code},
            )
        except Exception as exc:
            return IntegrationResult(success=False, error=str(exc))

    async def invoke_kernel(self, component: str, params: dict[str, Any]) -> IntegrationResult:
        """Invoke a specific kernel component."""
        params["component"] = component
        return await self.execute("invoke", params)

    async def list_kernels(self) -> IntegrationResult:
        """List all available kernel components."""
        return await self.execute("list_components", {})


# ---------------------------------------------------------------------------
# Data Center Commander Integration
# ---------------------------------------------------------------------------


class DataCenterCommanderIntegration(BaseIntegration):
    """
    Integration with Data Center Commander for DC lifecycle management.

    Capabilities:
      - Server provisioning and decommissioning
      - Capacity planning and monitoring
      - Power and cooling management
      - Network topology management
      - Incident management
    """

    @property
    def name(self) -> str:
        return "data_center_commander"

    async def execute(self, operation: str, params: dict[str, Any]) -> IntegrationResult:
        """Execute a data center operation."""
        try:
            client = await self._get_client()

            if operation == "provision":
                # Provision a new server or resource
                response = await client.post("/v1/servers/provision", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "decommission":
                # Decommission a server
                server_id = params.pop("server_id", None)
                if not server_id:
                    return IntegrationResult(success=False, error="server_id is required")
                response = await client.post(f"/v1/servers/{server_id}/decommission", json=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "capacity":
                # Get capacity information
                response = await client.get("/v1/capacity", params=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "power":
                # Get power consumption data
                response = await client.get("/v1/power", params=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "cooling":
                # Get cooling system status
                response = await client.get("/v1/cooling", params=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "network":
                # Get network topology
                response = await client.get("/v1/network/topology", params=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "incident":
                # Report or query an incident
                if params.get("action") == "report":
                    response = await client.post("/v1/incidents", json=params)
                else:
                    response = await client.get("/v1/incidents", params=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            elif operation == "servers":
                # List all servers
                response = await client.get("/v1/servers", params=params)
                response.raise_for_status()
                return IntegrationResult(success=True, data=response.json())

            else:
                return IntegrationResult(
                    success=False,
                    error=f"Unknown operation: {operation}",
                )

        except httpx.HTTPStatusError as exc:
            return IntegrationResult(
                success=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.text}",
                metadata={"status_code": exc.response.status_code},
            )
        except Exception as exc:
            return IntegrationResult(success=False, error=str(exc))

    async def get_capacity(self, datacenter: Optional[str] = None) -> IntegrationResult:
        """Get capacity information for a data center."""
        params = {}
        if datacenter:
            params["datacenter"] = datacenter
        return await self.execute("capacity", params)

    async def list_servers(self, status: Optional[str] = None) -> IntegrationResult:
        """List servers, optionally filtered by status."""
        params = {}
        if status:
            params["status"] = status
        return await self.execute("servers", params)


# ---------------------------------------------------------------------------
# Integration Registry
# ---------------------------------------------------------------------------


class IntegrationRegistry:
    """
    Registry and lifecycle manager for all integrations.

    Provides centralized access, health monitoring, and graceful shutdown.
    """

    def __init__(self) -> None:
        self._integrations: dict[str, BaseIntegration] = {}

    def register(self, integration: BaseIntegration) -> None:
        """Register an integration."""
        self._integrations[integration.name] = integration

    def get(self, name: str) -> Optional[BaseIntegration]:
        """Get an integration by name."""
        return self._integrations.get(name)

    def list_integrations(self) -> list[str]:
        """List all registered integration names."""
        return list(self._integrations.keys())

    async def health_check_all(self) -> dict[str, IntegrationHealth]:
        """Check health of all integrations."""
        results = {}
        for name, integration in self._integrations.items():
            results[name] = await integration.health_check()
        return results

    async def close_all(self) -> None:
        """Close all integration connections."""
        await asyncio.gather(
            *[integration.close() for integration in self._integrations.values()],
            return_exceptions=True,
        )

    @classmethod
    def create_default(
        cls,
        graph_swarm_url: str = "http://localhost:8001",
        grc_claw_url: str = "http://localhost:8002",
        apex_ull_url: str = "http://localhost:8003",
        dc_commander_url: str = "http://localhost:8004",
        api_key: Optional[str] = None,
    ) -> "IntegrationRegistry":
        """Create a registry with all default integrations configured."""
        registry = cls()
        registry.register(ApexGraphSwarmIntegration(graph_swarm_url, api_key))
        registry.register(GRCClawIntegration(grc_claw_url, api_key))
        registry.register(ApexULLIntegration(apex_ull_url, api_key))
        registry.register(DataCenterCommanderIntegration(dc_commander_url, api_key))
        return registry
