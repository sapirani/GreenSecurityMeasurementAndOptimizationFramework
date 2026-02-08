# SplunkResearch Configuration Guide

This directory contains all configuration files for the SplunkResearch project. All hardcoded values have been moved to these configuration files for easier management and security.

## Configuration Files

### 1. `default.yaml`
Contains all non-sensitive default values and parameters:
- **Reward parameters**: Alert thresholds, beta/gamma values, normalization factors
- **Action parameters**: Diversity factors, step size multipliers, thread pool configuration
- **Splunk settings**: Port numbers, pull intervals, default hosts
- **Elasticsearch settings**: Port configuration
- **State parameters**: Alert normalization factors
- **File paths**: Model paths, log file paths (with configurable base directory)

**✅ Safe to commit to version control**

### 2. `secrets.yaml`
Contains sensitive credentials and host-specific configuration:
- Splunk credentials (username, password, host)
- Elasticsearch credentials (username, password)
- Email configuration (for notifications)
- Actual host IP addresses

**⚠️ NEVER commit this file to version control! It's already in .gitignore**

### 3. `secrets.yaml.example`
Template showing the structure of secrets.yaml. Copy this file to create your own secrets.yaml.

**✅ Safe to commit - contains no actual secrets**

## Setup Instructions

### First Time Setup

1. **Copy the secrets template:**
   ```bash
   cd /home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/config
   cp secrets.yaml.example secrets.yaml
   ```

2. **Edit secrets.yaml with your actual credentials:**
   ```bash
   nano secrets.yaml
   ```

   Replace all `YOUR_*` placeholders with your actual values:
   - Splunk host, username, and password
   - Elasticsearch password
   - Email credentials (if using notifications)
   - Host IP addresses

3. **Verify .gitignore:**
   Ensure `SplunkResearch/config/secrets.yaml` is in your `.gitignore` file (it already should be).

### Configuration Override

The configuration system uses a two-layer approach:

1. **Default values** from `default.yaml` are loaded first
2. **Secrets** from `secrets.yaml` override any matching keys

This means you can override any default value in `secrets.yaml` if needed for your specific environment.

## Usage in Code

### Importing Configuration

```python
from config import config

# Get a configuration value with dot notation
value = config.get('reward.beta', default_value)

# Get a required value (raises error if missing)
value = config.get_required('splunk.host')

# Get an entire section
reward_config = config.get_section('reward')
```

### Examples

```python
# Reward parameters
beta = config.get('reward.beta', 0.33)
normalizer_factor = config.get('reward.normalizer_factor', 10)

# Splunk connection
splunk_port = config.get('splunk.port', 8089)
splunk_host = config.get('splunk.host')  # From secrets.yaml

# Elasticsearch credentials (MUST be in secrets.yaml)
es_password = config.get('elasticsearch.password')

# File paths
base_dir = config.get('paths.base_dir')
model_path = config.get('paths.models_all_rules_cpu')
```

## Security Best Practices

### ✅ DO:
- Keep `secrets.yaml` out of version control
- Use environment-specific secrets.yaml files for different deployments
- Regularly rotate passwords and update secrets.yaml
- Set restrictive file permissions on secrets.yaml: `chmod 600 secrets.yaml`
- Use the configuration system for ALL sensitive values

### ❌ DON'T:
- Commit secrets.yaml to git
- Share secrets.yaml via email or chat
- Hardcode credentials in Python files
- Store secrets in default.yaml (non-sensitive only!)

## Configuration Reference

### Reward Parameters
```yaml
reward:
  alert:
    threshold: -0.5              # Alert threshold for execution decisions
    beta: 0.33                   # Alert reward weighting factor
    gamma: 0.33                  # Distribution reward weighting factor
    normalizer_factor: 10        # Normalization factor for tanh
    tanh_scaling_factor: -2      # Scaling factor for line 516 in reward.py
    std: 13                      # Standard deviation for calculations
    epsilon: 0.00000001         # Numerical stability constant
```

### Action Parameters
```yaml
action:
  diversity_factor: 30          # Diversity factor for action space
  step_size_multiplier: 2000    # Step size calculation multiplier
  max_workers: 3                # Thread pool size (keep low 2-4)
  random_threshold: 10          # Random probability threshold
```

### Connection Settings
```yaml
splunk:
  port: 8089                    # Splunk management port
  default_host: "dt-splunk"     # Default hostname in queries
  binary_path: "/opt/splunk/bin/splunk"  # Path to Splunk binary

elasticsearch:
  port: 9200                    # Elasticsearch port
```

### File Paths
```yaml
paths:
  base_dir: "/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch"
  models_all_rules_cpu: "src/models_all_rules_cpu.joblib"
  cpu_model_template: "src/cpu_model_{rule}.joblib"
```

## Troubleshooting

### "Configuration key not found" Error
- Check if the key exists in `default.yaml` or `secrets.yaml`
- Verify the dot notation path is correct (e.g., `reward.beta` not `reward_beta`)
- For secrets, ensure `secrets.yaml` exists and is properly formatted

### "Secrets file not found" Warning
- Copy `secrets.yaml.example` to `secrets.yaml`
- Fill in your actual credentials

### "Elasticsearch password not configured" Warning
- Add the Elasticsearch password to `secrets.yaml`:
  ```yaml
  elasticsearch:
    password: "your_actual_password"
  ```

### Import Errors
- Ensure you're importing from the correct path
- The `config.py` module should be in `SplunkResearch/src/`
- Check that `sys.path` includes the src directory

## Migration Notes

### What Changed
All hardcoded values have been removed from:
- `reward.py` - All reward parameters, model paths, hosts
- `action.py` - Diversity factors, thread pool size, hosts
- `splunk_tools.py` - **CRITICAL**: Elasticsearch password removed, all paths and hosts
- `state.py` - Alert normalization factor, pickle file paths
- `energy_profile_final.py` - Log paths, hosts, port numbers

### Backward Compatibility
The code maintains backward compatibility through default values. If a config key is missing, the original hardcoded value is used as a fallback.

## Support

For issues or questions about configuration:
1. Check this README first
2. Verify your YAML syntax is correct
3. Check the logs for specific error messages
4. Ensure file permissions are correct on secrets.yaml

---

**Last Updated:** 2026-02-08
**Version:** 1.0
