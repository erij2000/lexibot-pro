import asyncio
import asyncpg

async def run():
    try:
        conn = await asyncpg.connect(
            host='127.0.0.1', 
            port=5432, 
            user='erij_user', 
            password='super_secret_erij_mdp', 
            database='lexibot_db'
        )
        print("✅ CONNEXION RÉUSSIE !")
        await conn.close()
    except Exception as e:
        print(f"❌ ÉCHEC : {e}")

if __name__ == "__main__":
    asyncio.run(run())