#!/usr/bin/env python3
"""
Test script to verify that the delivered_by functionality is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spoiled_broth.game import SpoiledBroth
from spoiled_broth.agent.base import Agent
from spoiled_broth.agent.intents import DeliveryIntent, ItemExchangeIntent, CuttingBoardIntent

def test_delivery_tracking():
    """Test that delivered_by field is properly set when delivery occurs."""
    
    # Create a game
    game = SpoiledBroth(map_nr=1)
    
    # Add an agent
    agent_id = "test_agent"
    game.add_agent(agent_id)
    agent = game.gameObjects[agent_id]
    
    # Find delivery tile
    delivery_tile = None
    for x in range(game.grid.width):
        for y in range(game.grid.height):
            tile = game.grid.tiles[x][y]
            if hasattr(tile, 'delivered_by'):
                delivery_tile = tile
                break
        if delivery_tile:
            break
    
    if not delivery_tile:
        print("❌ No delivery tile found!")
        return False
    
    print(f"✅ Found delivery tile at ({delivery_tile.slot_x}, {delivery_tile.slot_y})")
    
    # Give agent a salad to deliver
    agent.item = "tomato_salad"
    print(f"✅ Agent has item: {agent.item}")
    
    # Initial state
    print(f"Initial delivered_by: {delivery_tile.delivered_by}")
    
    # Directly set DeliveryIntent and process it
    agent.set_intents([DeliveryIntent(delivery_tile)])
    agent.update({}, 0.1)
    
    # Check if delivered_by was set
    print(f"After delivery, delivered_by: {delivery_tile.delivered_by}")
    
    if delivery_tile.delivered_by == agent_id:
        print("✅ SUCCESS: delivered_by field was properly set!")
        return True
    else:
        print("❌ FAILED: delivered_by field was not set correctly!")
        return False

def test_salad_creation_tracking():
    """Test that salad_by field is properly set when salad is created."""
    
    # Create a game
    game = SpoiledBroth(map_nr=1)
    
    # Add an agent
    agent_id = "test_agent"
    game.add_agent(agent_id)
    agent = game.gameObjects[agent_id]
    
    # Find counter tile
    counter_tile = None
    for x in range(game.grid.width):
        for y in range(game.grid.height):
            tile = game.grid.tiles[x][y]
            if hasattr(tile, 'salad_by'):
                counter_tile = tile
                break
        if counter_tile:
            break
    
    if not counter_tile:
        print("❌ No counter tile found!")
        return False
    
    print(f"✅ Found counter tile at ({counter_tile.slot_x}, {counter_tile.slot_y})")
    
    # Set up for salad creation: agent has cut item, counter has plate
    agent.item = "tomato_cut"
    counter_tile.item = "plate"
    print(f"✅ Agent has: {agent.item}, Counter has: {counter_tile.item}")
    
    # Initial state
    print(f"Initial salad_by: {counter_tile.salad_by}")
    
    # Directly set ItemExchangeIntent and process it
    agent.set_intents([ItemExchangeIntent(counter_tile)])
    agent.update({}, 0.1)
    
    # Check if salad_by was set
    print(f"After salad creation, salad_by: {counter_tile.salad_by}")
    print(f"Counter now has: {counter_tile.item}")
    
    if counter_tile.salad_by == agent_id and counter_tile.item == "tomato_salad":
        print("✅ SUCCESS: salad_by field was properly set!")
        return True
    else:
        print("❌ FAILED: salad_by field was not set correctly!")
        return False

def test_cutting_tracking():
    """Test that cut_by field is properly set when item is cut."""
    
    # Create a game
    game = SpoiledBroth(map_nr=1)
    
    # Add an agent
    agent_id = "test_agent"
    game.add_agent(agent_id)
    agent = game.gameObjects[agent_id]
    
    # Find cutting board tile
    cutting_board = None
    for x in range(game.grid.width):
        for y in range(game.grid.height):
            tile = game.grid.tiles[x][y]
            if hasattr(tile, 'cut_by'):
                cutting_board = tile
                break
        if cutting_board:
            break
    
    if not cutting_board:
        print("❌ No cutting board found!")
        return False
    
    print(f"✅ Found cutting board at ({cutting_board.slot_x}, {cutting_board.slot_y})")
    
    # Set up for cutting: put tomato on cutting board
    cutting_board.item = "tomato"
    cutting_board.cut_time_accumulated = 3.0  # Fully cut
    print(f"✅ Cutting board has: {cutting_board.item}, cut stage: {cutting_board.cut_stage}")
    
    # Initial state
    print(f"Initial cut_by: {cutting_board.cut_by}")
    
    # Directly set CuttingBoardIntent and process it
    agent.set_intents([CuttingBoardIntent(cutting_board)])
    agent.update({}, 0.1)
    
    # Check if cut_by was set
    print(f"After cutting, cut_by: {cutting_board.cut_by}")
    print(f"Agent now has: {agent.item}")
    
    if cutting_board.cut_by == agent_id and agent.item == "tomato_cut":
        print("✅ SUCCESS: cut_by field was properly set!")
        return True
    else:
        print("❌ FAILED: cut_by field was not set correctly!")
        return False

if __name__ == "__main__":
    print("Testing delivery tracking functionality...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    print("\n1. Testing delivery tracking...")
    if test_delivery_tracking():
        success_count += 1
    
    print("\n2. Testing salad creation tracking...")
    if test_salad_creation_tracking():
        success_count += 1
    
    print("\n3. Testing cutting tracking...")
    if test_cutting_tracking():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The tracking functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.") 