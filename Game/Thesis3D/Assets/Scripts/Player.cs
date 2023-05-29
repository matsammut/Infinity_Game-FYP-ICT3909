using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;


public class Player : MonoBehaviour
{
    public int player_movement_speed ;
    public int player_jump_speed ;
    private Rigidbody rb2d;
    private float movementX;
    private float movementY;
    public int jump_charge_remaining = 1;
    public int max_jumps = 1;

    private bool isStarted = false;

    private string GROUND_TAG = "Ground";

    // Start is called before the first frame update
    void Start()
    {
        rb2d = this.gameObject.GetComponent<Rigidbody>();
        Time.timeScale = 0f;
    }

    // Update is called once per frame
     void Update()
    {

        //Press to start
        if (Input.GetMouseButtonDown(0) && isStarted == false)
        {
            Time.timeScale = 1f;
            // startText.gameObject.SetActive(false);
        }

            movementX = Input.GetAxisRaw("Horizontal");
            movementY = Input.GetAxisRaw("Vertical");
            if (Input.GetKey(KeyCode.A)) {
                transform.Translate(Vector3.right * Time.deltaTime * player_movement_speed);
                transform.localScale = new Vector3(1f,1f,1f);
            }

            if (Input.GetKey(KeyCode.D)) {
                transform.Translate(Vector3.left * Time.deltaTime * player_movement_speed);
                transform.localScale = new Vector3(-1f,1f,1f);
            }
            if (jump_charge_remaining > 0) {
              if (Input.GetKeyDown(KeyCode.Space)) {
                jump_charge_remaining --;
                rb2d.AddForce(Vector2.up*player_jump_speed, ForceMode.Impulse);
              }
            }

    }

    void OnCollisionEnter(Collision collision)
    {

      if (collision.gameObject.CompareTag(GROUND_TAG))
          {
            jump_charge_remaining = max_jumps;
          }
    }


}
