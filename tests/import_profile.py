import asyncio

from pyinstrument import Profiler


async def main():
    p = Profiler(async_mode="disabled")

    with p:
        import fusion_bench

    p.print()


asyncio.run(main())
