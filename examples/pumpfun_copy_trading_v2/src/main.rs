use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use serde::Deserialize;
use sol_trade_sdk::common::{
    fast_fn::get_associated_token_address_with_program_id_fast_use_seed, TradeConfig,
};
use sol_trade_sdk::TradeTokenType;
use sol_trade_sdk::{
    common::AnyResult,
    swqos::SwqosConfig,
    trading::{
        core::params::{DexParamEnum, PumpFunParams},
        factory::DexType,
    },
    SolanaTrade,
};
use solana_commitment_config::CommitmentConfig;
use solana_sdk::hash::Hash;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::read_keypair_file;
use solana_sdk::signer::Signer;

#[derive(Deserialize, Debug)]
struct TradeEventTemplate {
    token: String,
    protocol_data: ProtocolData,
    token_info: TokenInfo,
    bonding_curve_data: BondingCurveData,
    trade_details: TradeDetails,
}

#[derive(Deserialize, Debug)]
struct ProtocolData {
    bonding_curve: String,
    associated_bonding_curve: String,
    creator_vault: String,
    token_program: String,
}

#[derive(Deserialize, Debug)]
struct TradeDetails {
    fees: Fees,
}

#[derive(Deserialize, Debug)]
struct Fees {
    fee_recipient: String,
}

#[derive(Deserialize, Debug)]
struct TokenInfo {
    creator: String,
}

#[derive(Deserialize, Debug)]
struct BondingCurveData {
    virtual_sol_reserves: u64,
    virtual_token_reserves: u64,
    real_sol_reserves: u64,
    real_token_reserves: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading trade event from JSON template...");

    // 1. Load the trade event template
    let template_path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/pumpfun-token-trade-event-template.json");
    let template_content = std::fs::read_to_string(template_path)?;
    let template: TradeEventTemplate = serde_json::from_str(&template_content)?;

    println!("Copy trading token: {}", template.token);

    // 2. Prepare params from template
    let mint = Pubkey::from_str(&template.token)?;
    let protocol = &template.protocol_data;
    let bonding = &template.bonding_curve_data;

    let pumpfun_params = PumpFunParams::from_trade(
        Pubkey::from_str(&protocol.bonding_curve)?,
        Pubkey::from_str(&protocol.associated_bonding_curve)?,
        mint,
        Pubkey::from_str(&template.token_info.creator)?,
        Pubkey::from_str(&protocol.creator_vault)?,
        bonding.virtual_token_reserves,
        bonding.virtual_sol_reserves,
        bonding.real_token_reserves,
        bonding.real_sol_reserves,
        None,
        Pubkey::from_str(&template.trade_details.fees.fee_recipient)?,
        Pubkey::from_str(&protocol.token_program)?,
    );

    // 3. Initialize background blockhash fetcher
    println!("🚀 Starting background blockhash fetcher (every 5s)...");
    let cached_hash = Arc::new(RwLock::new(None));
    let cached_hash_clone = cached_hash.clone();

    // Spawn background task to fetch blockhash every 5 seconds
    tokio::spawn(async move {
        // Use a temporary client for fetching
        if let Ok(client) = create_solana_trade_client().await {
            loop {
                match client.rpc.get_latest_blockhash().await {
                    Ok(hash) => {
                        let mut lock = cached_hash_clone.write().await;
                        *lock = Some(hash);
                        // println!("🔄 Background hash updated: {:?}", hash);
                    }
                    Err(e) => eprintln!("❌ Background hash fetch failed: {:?}", e),
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    });

    // Wait a moment for the first hash
    tokio::time::sleep(Duration::from_secs(1)).await;

    // 4. Execute copy trade in a 15-second loop
    println!("🧪 Starting 15-second performance test loop...");
    let start_time = Instant::now();
    let mut is_first_run = true;
    let mut counter = 0u64;

    while start_time.elapsed() < Duration::from_secs(15) {
        println!("\n--- Iteration (Elapsed: {:?}) ---", start_time.elapsed());

        // For the first run, we pass None to force a manual RPC call
        // For subsequent runs, we pass the cached hash from memory
        let hash_to_use = if is_first_run {
            None
        } else {
            let lock = cached_hash.read().await;
            *lock
        };

        if let Err(err) =
            pumpfun_copy_trade_with_params(mint, pumpfun_params.clone(), hash_to_use, counter).await
        {
            eprintln!("Error in copy trade iteration: {:?}", err);
        }

        is_first_run = false;
        counter += 1;
        // Small delay between iterations to avoid overwhelming local node
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    println!("\n🧹 Checking for remaining tokens to clean up...");
    // Force a final sell by calling the trade function one last time with a large salt
    // (In a real scenario, we might want a dedicated cleanup_sell function)
    if let Err(err) =
        pumpfun_copy_trade_with_params(mint, pumpfun_params.clone(), None, counter + 1000).await
    {
        // It's okay if this fails (e.g. no tokens left), we just want to attempt a final sell
        eprintln!("Final cleanup attempt finished: {:?}", err);
    }

    println!("\n✅ 15-second test loop finished.");
    Ok(())
}

/// Create SolanaTrade client
/// Initializes a new SolanaTrade client with local RPC and configured payer
async fn create_solana_trade_client() -> AnyResult<SolanaTrade> {
    println!("🚀 Initializing SolanaTrade client with local RPC...");

    // Load keypair from /home/jvelasco/.config/solana/id.json
    let keypair_path = "/home/jvelasco/.config/solana/id.json";
    let payer = read_keypair_file(keypair_path)
        .map_err(|e| anyhow::anyhow!("Failed to read keypair from {}: {:?}", keypair_path, e))?;

    let rpc_url = "http://localhost:8899".to_string();
    let commitment = CommitmentConfig::confirmed();
    let swqos_configs: Vec<SwqosConfig> = vec![SwqosConfig::Default(rpc_url.clone())];
    let trade_config = TradeConfig::new(rpc_url, swqos_configs, commitment);
    let solana_trade = SolanaTrade::new(Arc::new(payer), trade_config).await;

    println!("✅ SolanaTrade client initialized successfully!");
    Ok(solana_trade)
}

/// PumpFun copy trade execution
async fn pumpfun_copy_trade_with_params(
    mint_pubkey: Pubkey,
    pump_params: PumpFunParams,
    cached_hash: Option<Hash>,
    salt: u64,
) -> AnyResult<()> {
    println!("Testing PumpFun trading for mint: {}...", mint_pubkey);

    let client = create_solana_trade_client().await?;
    let slippage_basis_points = Some(5000);

    // Determine which blockhash to use
    let hash_get_start = Instant::now();
    let (recent_blockhash, is_cached) = match cached_hash {
        Some(hash) => (hash, true),
        None => (client.rpc.get_latest_blockhash().await?, false),
    };
    let hash_get_duration = hash_get_start.elapsed();

    println!(
        "{} Blockhash retrieval time: {}ms ({}µs)",
        if is_cached { "💡 [Memory]" } else { "🌐 [RPC]" },
        hash_get_duration.as_millis(),
        hash_get_duration.as_micros()
    );

    let gas_fee_strategy = sol_trade_sdk::common::GasFeeStrategy::new();
    gas_fee_strategy.set_global_fee_strategy(150000, 150000, 500000, 500000, 0.001, 0.001);

    // Check if the mint ATA already exists
    let payer = client.payer.pubkey();
    let token_account = get_associated_token_address_with_program_id_fast_use_seed(
        &payer,
        &mint_pubkey,
        &pump_params.token_program,
        client.use_seed_optimize,
    );
    let ata_exists = client.rpc.get_account(&token_account).await.is_ok();
    if ata_exists {
        println!("💡 Token account already exists, skipping creation.");
    }

    // Buy tokens
    println!("Buying tokens from PumpFun (Salt: {})...", salt);
    let buy_sol_amount = 501_844_400 + salt; // Use salt to ensure unique transaction signature

    let buy_params = sol_trade_sdk::TradeBuyParams {
        dex_type: DexType::PumpFun,
        input_token_type: TradeTokenType::SOL,
        mint: mint_pubkey,
        input_token_amount: buy_sol_amount,
        slippage_basis_points,
        recent_blockhash: Some(recent_blockhash),
        extension_params: DexParamEnum::PumpFun(pump_params.clone()),
        address_lookup_table_account: None,
        wait_transaction_confirmed: true,
        create_input_token_ata: false,
        close_input_token_ata: false,
        create_mint_ata: !ata_exists,
        durable_nonce: None,
        fixed_output_token_amount: None,
        gas_fee_strategy: gas_fee_strategy.clone(),
        simulate: false,
    };

    let buy_start = Instant::now();
    let (success, signatures, error) = match client.buy(buy_params).await {
        Ok(res) => res,
        Err(e) => {
            println!("❌ Buy transaction RPC error: {:?}", e);
            return Err(e);
        }
    };
    let buy_duration = buy_start.elapsed();

    if !success {
        println!(
            "❌ Buy transaction failed on-chain: signatures={:?}, error={:?}",
            signatures, error
        );
        return Err(anyhow::anyhow!("Buy transaction failed on-chain"));
    }

    println!("✅ Buy transaction successful: {:?}", signatures);
    println!(
        "⏱️ Buy TOTAL execution time (Wait for Confirmation: true): {}ms ({}µs)",
        buy_duration.as_millis(),
        buy_duration.as_micros()
    );

    // Sell tokens
    println!("Selling tokens from PumpFun...");

    let rpc = client.rpc.clone();
    let payer = client.payer.pubkey();
    let account = get_associated_token_address_with_program_id_fast_use_seed(
        &payer,
        &mint_pubkey,
        &pump_params.token_program,
        client.use_seed_optimize,
    );

    // Wait a bit for the buy to be reflected in some local nodes if necessary,
    // but usually confirmed is enough.
    println!("Checking token balance for account: {}...", account);

    let mut balance_opt = None;
    for i in 0..10 {
        match rpc.get_token_account_balance(&account).await {
            Ok(bal) => {
                balance_opt = Some(bal);
                break;
            }
            Err(e) => {
                if e.to_string().contains("-32602") || e.to_string().contains("not found") {
                    println!("Token account not found yet, retrying... ({}/10)", i + 1);
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                } else {
                    return Err(e.into());
                }
            }
        }
    }

    let balance = balance_opt.ok_or_else(|| {
        anyhow::anyhow!(
            "Token account {} not found after retries. The buy transaction likely failed or didn't land on your local RPC.",
            account
        )
    })?;

    println!("Balance: {:?}", balance);
    let amount_token = balance.amount.parse::<u64>().unwrap();

    if amount_token == 0 {
        println!("No tokens to sell. Exiting.");
        return Ok(());
    }

    println!("Selling {} tokens", amount_token);

    let mut sell_pump_params = pump_params.clone();
    sell_pump_params.close_token_account_when_sell = Some(false);

    let sell_params = sol_trade_sdk::TradeSellParams {
        dex_type: DexType::PumpFun,
        output_token_type: TradeTokenType::SOL,
        mint: mint_pubkey,
        input_token_amount: amount_token,
        slippage_basis_points,
        recent_blockhash: Some(recent_blockhash),
        with_tip: true,
        extension_params: DexParamEnum::PumpFun(sell_pump_params),
        address_lookup_table_account: None,
        wait_transaction_confirmed: true,
        create_output_token_ata: false,
        close_output_token_ata: false,
        close_mint_token_ata: true,
        durable_nonce: None,
        fixed_output_token_amount: None,
        gas_fee_strategy,
        simulate: false,
    };

    let sell_start = Instant::now();
    let (sell_success, sell_sigs, sell_err) = client.sell(sell_params).await?;
    let sell_duration = sell_start.elapsed();

    if sell_success {
        println!("✅ Sell transaction successful: {:?}", sell_sigs);
    } else {
        println!(
            "❌ Sell transaction failed on-chain: signatures={:?}, error={:?}",
            sell_sigs, sell_err
        );
    }

    println!(
        "⏱️ Sell execution time: {}ms ({}µs)",
        sell_duration.as_millis(),
        sell_duration.as_micros()
    );

    Ok(())
}
