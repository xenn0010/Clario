"""Simple integration check for Weaviate and Neo4j services."""

import asyncio
import pathlib
import sys
from pprint import pprint

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.services.vector.weaviate_service import WeaviateService
from app.services.graph.neo4j_service import Neo4jService


async def main() -> None:
    weaviate_service = await WeaviateService.initialize()
    print("Weaviate connected:", weaviate_service.is_connected)

    neo_service = await Neo4jService.initialize()
    stats = await neo_service.get_database_stats()
    print("Neo4j stats summary:")
    pprint(stats)

    await weaviate_service.cleanup()
    await neo_service.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
