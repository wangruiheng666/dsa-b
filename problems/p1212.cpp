#include <iostream>
#include <vector>
#include <stack>
#include <numeric>   // Not strictly needed
#include <algorithm> // Not strictly needed

using namespace std;

// adj: adjacency list
// sequence: the sequence to check
// n_vertices: number of vertices
bool is_dfs_sequence(const vector<vector<int>>& adj, const vector<int>& sequence, int n_vertices) {
    // Problem constraints: 0 < n <= 16. So n_vertices > 0.
    // sequence.size() is guaranteed to be n_vertices.

    vector<bool> visited(n_vertices, false);
    stack<int> s;         
    int seq_idx = 0;      

    while (seq_idx < n_vertices) { 
        if (s.empty()) {
            int start_node_of_component = sequence[seq_idx];
            if (visited[start_node_of_component]) {
                return false; // Should not happen if sequence is a permutation of 0..n-1 and valid
            }
            s.push(start_node_of_component);
            visited[start_node_of_component] = true;
            seq_idx++;
        }
        
        while(!s.empty()){
            int u = s.top();
            bool advanced_by_finding_child = false;

            if (seq_idx < n_vertices) {
                int target_node = sequence[seq_idx];

                if (visited[target_node]) {
                    return false; // Next node in sequence cannot be already visited
                }

                bool is_target_neighbor_of_u = false;
                for (int neighbor_of_u : adj[u]) {
                    if (neighbor_of_u == target_node) {
                        is_target_neighbor_of_u = true;
                        break;
                    }
                }

                if (is_target_neighbor_of_u) {
                    visited[target_node] = true;
                    s.push(target_node);
                    seq_idx++;
                    advanced_by_finding_child = true; 
                                                
                } else { // target_node is NOT an unvisited neighbor of u
                    bool u_has_any_unvisited_neighbor = false;
                    for (int neighbor_of_u : adj[u]) {
                        if (!visited[neighbor_of_u]) {
                            u_has_any_unvisited_neighbor = true;
                            break;
                        }
                    }

                    if (u_has_any_unvisited_neighbor) {
                        // u could explore other paths, but sequence doesn't allow. Invalid.
                        return false;
                    } else {
                        // u has no other unvisited children. Must backtrack.
                        s.pop();
                        // advanced_by_finding_child remains false.
                    }
                }
            } else { // seq_idx == n_vertices. All sequence nodes processed.
                s.pop(); // Backtrack to empty the stack.
                // advanced_by_finding_child remains false.
            }
            
            if (advanced_by_finding_child) {
                // If u found a child, the new child is s.top().
                // The inner while loop will continue, processing this new s.top().
                // No explicit 'continue' needed here for the inner loop.
            } else {
                // If u did not find a child as per sequence (either popped or returned false),
                // this iteration for u is done.
                // If u was popped, inner loop continues with parent or terminates if stack empty.
            }
        } 
    } 

    // If loop completes, seq_idx == n_vertices. Stack should be empty.
    return true; 
}


int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    cin >> n >> m;

    vector<vector<int>> adj(n);
    for (int i = 0; i < m; ++i) {
        int u, v_node; 
        cin >> u >> v_node;
        adj[u].push_back(v_node);
        adj[v_node].push_back(u);
    }

    int k;
    cin >> k;
    for (int i = 0; i < k; ++i) {
        vector<int> sequence_to_check(n);
        for (int j = 0; j < n; ++j) {
            cin >> sequence_to_check[j];
        }
        if (is_dfs_sequence(adj, sequence_to_check, n)) {
            cout << "YES\n";
        } else {
            cout << "NO\n";
        }
    }

    return 0;
}