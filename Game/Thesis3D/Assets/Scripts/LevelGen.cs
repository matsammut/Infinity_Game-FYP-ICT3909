using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LevelGen : MonoBehaviour
{
    public GameObject[] levelSections; // Array of level section prefabs
    private float spawnPoint = 20f; // The point where new level sections will be spawned
    public float screenWidth; // The width of the screen in world units

    private void OnCollisionEnter(Collision collision) {       
    
        if (collision.gameObject.CompareTag("Player"))
          {
            // Choose a random level section prefab
            int randomIndex = Random.Range(0, levelSections.Length);
            Vector3 pos = new (spawnPoint,0f,0f);
            GameObject newSection = Instantiate(levelSections[randomIndex],pos , Quaternion.identity);

            // move the spawnpoint by 10
            spawnPoint += 20;

            transform.position = new Vector3(transform.position.x + 20f, transform.position.y, transform.position.z);

        }
    }
}
